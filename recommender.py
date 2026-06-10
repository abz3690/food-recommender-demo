from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize

RANDOM_STATE = 42
BEST_WEIGHTS = (0.50, 0.35, 0.15)

REQUIRED_FOOD_SHEETS = {"foods", "places", "place_food_map"}
REQUIRED_RATING_SHEETS = {"users", "user_ratings", "rated_only"}

DEMO_EXCLUDE_KEYWORDS = {
    "bánh bao", "bánh tôm", "sữa chua", "nếp cẩm", "chuối nếp",
    "kem", "chè", "tráng miệng", "tiramisu", "croissant", "bánh ngọt",
}

ALLOWED_MAIN_TYPES = {
    "main_dish", "rice_dish", "noodle_dish", "noodle_soup", "hotpot",
    "grilled_dish", "fried_dish", "curry_stew", "salad", "sushi",
    "ramen", "steak",
}


@dataclass
class RecommendationSystem:
    foods: pd.DataFrame
    places: pd.DataFrame
    place_food_map: pd.DataFrame
    users: pd.DataFrame
    ratings: pd.DataFrame
    train_ratings: pd.DataFrame
    foods_model: pd.DataFrame
    food_matrix: sparse.spmatrix
    food_id_to_idx: dict[str, int]
    cf_predictions: pd.DataFrame
    item_mean: pd.Series
    global_mean: float
    popularity_map: pd.Series
    weights: tuple[float, float, float] = BEST_WEIGHTS

    @property
    def eligible_users(self) -> list[str]:
        return sorted(self.ratings["user_id"].astype(str).unique().tolist())

    @property
    def available_cuisines(self) -> list[str]:
        if "cuisine_group" not in self.foods_model.columns:
            return []
        values = self.foods_model["cuisine_group"].dropna().astype(str)
        return sorted(v for v in values.unique() if v.strip() and v.lower() != "unknown")

    def _minmax(self, values: pd.Series) -> pd.Series:
        values = pd.Series(values, dtype=float)
        if values.empty or np.isclose(values.max(), values.min()):
            return pd.Series(0.0, index=values.index)
        return (values - values.min()) / (values.max() - values.min())

    def content_scores_for_user(self, user_id: str) -> pd.Series:
        user_id = str(user_id)
        history = self.train_ratings[
            (self.train_ratings["user_id"] == user_id)
            & (self.train_ratings["food_id"].isin(self.food_id_to_idx))
        ].copy()
        food_ids = self.foods_model["food_id"].astype(str)
        if history.empty:
            return pd.Series(0.0, index=food_ids)

        indices = [self.food_id_to_idx[fid] for fid in history["food_id"]]
        weights = history["rating"].to_numpy(dtype=float) - 3.0
        if np.allclose(weights, 0):
            weights = np.ones_like(weights)

        profile = self.food_matrix[indices].multiply(weights[:, None]).sum(axis=0)
        profile = normalize(sparse.csr_matrix(profile))
        scores = cosine_similarity(profile, self.food_matrix).ravel()
        return pd.Series(scores, index=food_ids)

    def user_score_table(self, user_id: str) -> pd.DataFrame:
        alpha, beta, gamma = self.weights
        user_id = str(user_id)
        result = self.foods_model[["food_id", "dish_name"]].copy()
        result["food_id"] = result["food_id"].astype(str)
        all_ids = result["food_id"].tolist()

        content = self.content_scores_for_user(user_id).reindex(all_ids).fillna(0)
        if user_id in self.cf_predictions.index:
            cf = self.cf_predictions.loc[user_id].reindex(all_ids)
            cf = cf.fillna(self.item_mean.reindex(all_ids)).fillna(self.global_mean)
        else:
            cf = self.item_mean.reindex(all_ids).fillna(self.global_mean)

        pop = self.popularity_map.reindex(all_ids).fillna(0)
        result["cf_score"] = cf.to_numpy()
        result["content_score"] = content.to_numpy()
        result["popularity_score"] = pop.to_numpy()
        result["hybrid_score"] = (
            alpha * self._minmax(result["cf_score"]).to_numpy()
            + beta * self._minmax(result["content_score"]).to_numpy()
            + gamma * self._minmax(result["popularity_score"]).to_numpy()
        )
        return result

    def new_user_score_table(
        self,
        liked_food_ids: Sequence[str] | None = None,
        disliked_food_ids: Sequence[str] | None = None,
        preferred_cuisines: Sequence[str] | None = None,
        preferred_dish_types: Sequence[str] | None = None,
        preferred_prices: Sequence[str] | None = None,
        preferred_spicy_levels: Sequence[str] | None = None,
        excluded_keywords: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Create a cold-start profile without requiring a trained user_id."""
        liked = [str(x) for x in (liked_food_ids or []) if str(x) in self.food_id_to_idx]
        disliked = [str(x) for x in (disliked_food_ids or []) if str(x) in self.food_id_to_idx]

        result = self.foods_model[["food_id", "dish_name"]].copy()
        result["food_id"] = result["food_id"].astype(str)

        seeds = liked + disliked
        if seeds:
            indices = [self.food_id_to_idx[fid] for fid in seeds]
            weights = np.array([1.0] * len(liked) + [-0.8] * len(disliked), dtype=float)
            profile = self.food_matrix[indices].multiply(weights[:, None]).sum(axis=0)
            profile = sparse.csr_matrix(profile)
            if np.linalg.norm(profile.toarray()) > 0:
                profile = normalize(profile)
                content = cosine_similarity(profile, self.food_matrix).ravel()
            else:
                content = np.zeros(len(self.foods_model))
        else:
            content = np.zeros(len(self.foods_model))

        explicit = np.zeros(len(self.foods_model), dtype=float)
        parts = 0
        filters = [
            ("cuisine_group", preferred_cuisines),
            ("dish_type", preferred_dish_types),
            ("price_range", preferred_prices),
            ("spicy_level", preferred_spicy_levels),
        ]
        for col, values in filters:
            if values and col in self.foods_model.columns:
                allowed = {str(v).strip().lower() for v in values}
                explicit += self.foods_model[col].astype(str).str.strip().str.lower().isin(allowed).astype(float).to_numpy()
                parts += 1
        if parts:
            explicit /= parts

        # Penalize foods containing ingredients/styles the user wants to avoid.
        avoid_penalty = np.zeros(len(self.foods_model), dtype=float)
        if excluded_keywords:
            searchable_cols = [c for c in ["dish_name", "main_ingredient", "dish_type", "tags", "description"] if c in self.foods_model.columns]
            searchable = self.foods_model[searchable_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
            for keyword in excluded_keywords:
                key = str(keyword).strip().lower()
                if key:
                    avoid_penalty = np.maximum(avoid_penalty, searchable.str.contains(key, regex=False).astype(float).to_numpy())

        pop = self.popularity_map.reindex(result["food_id"]).fillna(0).to_numpy(dtype=float)
        content_norm = self._minmax(pd.Series(content)).to_numpy()
        pop_norm = self._minmax(pd.Series(pop)).to_numpy()
        if seeds:
            hybrid = 0.65 * content_norm + 0.25 * explicit + 0.10 * pop_norm
        elif parts:
            hybrid = 0.75 * explicit + 0.25 * pop_norm
        else:
            hybrid = pop_norm

        result["content_score"] = content
        result["preference_score"] = explicit
        result["popularity_score"] = pop
        result["hybrid_score"] = np.clip(hybrid - 0.95 * avoid_penalty, 0, 1)
        result["avoid_penalty"] = avoid_penalty
        return result

    def recommend_new_user(
        self,
        liked_food_ids: Sequence[str] | None = None,
        disliked_food_ids: Sequence[str] | None = None,
        preferred_cuisines: Sequence[str] | None = None,
        preferred_dish_types: Sequence[str] | None = None,
        preferred_prices: Sequence[str] | None = None,
        preferred_spicy_levels: Sequence[str] | None = None,
        excluded_keywords: Sequence[str] | None = None,
        top_k: int = 10,
        main_meal_only: bool = True,
    ) -> pd.DataFrame:
        scores = self.new_user_score_table(
            liked_food_ids=liked_food_ids,
            disliked_food_ids=disliked_food_ids,
            preferred_cuisines=preferred_cuisines,
            preferred_dish_types=preferred_dish_types,
            preferred_prices=preferred_prices,
            preferred_spicy_levels=preferred_spicy_levels,
            excluded_keywords=excluded_keywords,
        )
        # For cold-start users, cuisine/type/price/spicy inputs should guide ranking,
        # not act as hard filters. Hard filtering easily produces no result when
        # users pick a combination that is sparse in the demo dataset.
        scores = self._filter_scores(
            scores, self.build_candidate_catalog(main_meal_only), preferred_cuisines=None
        )
        if scores.empty and main_meal_only:
            # Fallback: allow all dish types instead of returning an empty screen.
            scores = self.new_user_score_table(
                liked_food_ids=liked_food_ids,
                disliked_food_ids=disliked_food_ids,
                preferred_cuisines=preferred_cuisines,
                preferred_dish_types=preferred_dish_types,
                preferred_prices=preferred_prices,
                preferred_spicy_levels=preferred_spicy_levels,
                excluded_keywords=excluded_keywords,
            )
            scores = self._filter_scores(scores, self.build_candidate_catalog(False), preferred_cuisines=None)
        excluded = set(map(str, (liked_food_ids or []))) | set(map(str, (disliked_food_ids or [])))
        scores = scores[~scores["food_id"].isin(excluded)]
        metadata_cols = [c for c in [
            "food_id", "dish_name", "cuisine_group", "dish_type",
            "main_ingredient", "price_range", "spicy_level", "healthy_score",
        ] if c in self.foods_model.columns]
        metadata = self.foods_model[metadata_cols].copy()
        metadata["food_id"] = metadata["food_id"].astype(str)
        result = scores.sort_values("hybrid_score", ascending=False).head(top_k)
        return result.drop(columns=["dish_name"], errors="ignore").merge(metadata, on="food_id", how="left")

    def group_recommend_new_users(
        self,
        profiles: Sequence[dict],
        strategy: str = "average",
        top_k: int = 10,
        main_meal_only: bool = True,
    ) -> pd.DataFrame:
        tables = []
        excluded = set()
        for idx, profile in enumerate(profiles, start=1):
            table = self.new_user_score_table(**profile).set_index("food_id")["hybrid_score"]
            tables.append(table.rename(f"Người {idx}"))
            excluded.update(map(str, profile.get("liked_food_ids", []) or []))
            excluded.update(map(str, profile.get("disliked_food_ids", []) or []))

        matrix = pd.concat(tables, axis=1).fillna(0)
        allowed = self.build_candidate_catalog(main_meal_only)
        matrix = matrix.loc[matrix.index.intersection(allowed)]
        if matrix.empty and main_meal_only:
            allowed = self.build_candidate_catalog(False)
            matrix = pd.concat(tables, axis=1).fillna(0)
            matrix = matrix.loc[matrix.index.intersection(allowed)]
        if strategy == "average":
            group_score = matrix.mean(axis=1)
        elif strategy == "least_misery":
            group_score = matrix.min(axis=1)
        elif strategy == "most_pleasure":
            group_score = matrix.max(axis=1)
        else:
            raise ValueError("Chiến lược nhóm không hợp lệ")

        ranking = group_score[~group_score.index.isin(excluded)].sort_values(ascending=False).head(top_k)
        result = ranking.rename("group_score").reset_index()
        metadata_cols = [c for c in ["food_id", "dish_name", "cuisine_group", "dish_type", "price_range"] if c in self.foods_model]
        metadata = self.foods_model[metadata_cols].copy()
        metadata["food_id"] = metadata["food_id"].astype(str)
        result = result.merge(metadata, on="food_id", how="left")
        for col in matrix.columns:
            result[f"score_{col}"] = result["food_id"].map(matrix[col])
        score_cols = [c for c in result.columns if c.startswith("score_")]
        result["minimum_score"] = result[score_cols].min(axis=1)
        result["fairness_gap"] = result[score_cols].max(axis=1) - result[score_cols].min(axis=1)
        return result

    def build_candidate_catalog(self, main_meal_only: bool = False) -> set[str]:
        catalog = self.foods_model[["food_id", "dish_name"]].copy()
        catalog["food_id"] = catalog["food_id"].astype(str)
        if not main_meal_only:
            return set(catalog["food_id"])

        names = catalog["dish_name"].fillna("").astype(str).str.lower()
        mask = ~names.apply(lambda x: any(keyword in x for keyword in DEMO_EXCLUDE_KEYWORDS))
        if "dish_type" in self.foods_model.columns:
            types = self.foods_model["dish_type"].fillna("").astype(str).str.lower()
            mask &= types.isin(ALLOWED_MAIN_TYPES)
        return set(catalog.loc[mask, "food_id"])

    def _filter_scores(
        self,
        scores: pd.DataFrame,
        candidate_catalog: Iterable[str] | None,
        preferred_cuisines: Sequence[str] | None,
    ) -> pd.DataFrame:
        out = scores
        if candidate_catalog is not None:
            out = out[out["food_id"].isin(set(map(str, candidate_catalog)))]
        if preferred_cuisines and "cuisine_group" in self.foods_model.columns:
            allowed = {str(v).strip().lower() for v in preferred_cuisines}
            cuisine = self.foods_model[["food_id", "cuisine_group"]].copy()
            cuisine["food_id"] = cuisine["food_id"].astype(str)
            cuisine["_cuisine"] = cuisine["cuisine_group"].astype(str).str.strip().str.lower()
            allowed_ids = set(cuisine.loc[cuisine["_cuisine"].isin(allowed), "food_id"])
            out = out[out["food_id"].isin(allowed_ids)]
        return out

    def recommend_user(
        self,
        user_id: str,
        top_k: int = 10,
        preferred_cuisines: Sequence[str] | None = None,
        main_meal_only: bool = False,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        scores = self.user_score_table(user_id)
        scores = self._filter_scores(
            scores,
            self.build_candidate_catalog(main_meal_only),
            preferred_cuisines,
        )
        if scores.empty and preferred_cuisines:
            # Relax cuisine filter if selected cuisines are too narrow.
            scores = self._filter_scores(
                self.user_score_table(user_id),
                self.build_candidate_catalog(main_meal_only),
                preferred_cuisines=None,
            )
        if scores.empty and main_meal_only:
            scores = self._filter_scores(
                self.user_score_table(user_id),
                self.build_candidate_catalog(False),
                preferred_cuisines=None,
            )
        if exclude_seen:
            seen = set(self.train_ratings.loc[
                self.train_ratings["user_id"] == str(user_id), "food_id"
            ])
            scores = scores[~scores["food_id"].isin(seen)]
            if scores.empty:
                # If the user has rated almost every candidate, show the best candidates anyway.
                scores = self._filter_scores(
                    self.user_score_table(user_id),
                    self.build_candidate_catalog(main_meal_only),
                    preferred_cuisines=None,
                )

        metadata_cols = [c for c in [
            "food_id", "dish_name", "cuisine_group", "dish_type",
            "main_ingredient", "price_range", "spicy_level", "healthy_score",
        ] if c in self.foods_model.columns]
        metadata = self.foods_model[metadata_cols].copy()
        metadata["food_id"] = metadata["food_id"].astype(str)
        result = scores.sort_values("hybrid_score", ascending=False).head(top_k)
        result = result.drop(columns=["dish_name"], errors="ignore").merge(metadata, on="food_id", how="left")
        return result

    def group_recommend(
        self,
        user_ids: Sequence[str],
        strategy: str = "average",
        top_k: int = 10,
        preferred_cuisines: Sequence[str] | None = None,
        main_meal_only: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        tables = []
        seen = set()
        for user_id in user_ids:
            table = self.user_score_table(str(user_id)).set_index("food_id")["hybrid_score"]
            tables.append(table.rename(str(user_id)))
            seen.update(self.train_ratings.loc[
                self.train_ratings["user_id"] == str(user_id), "food_id"
            ])

        matrix = pd.concat(tables, axis=1).fillna(0)
        allowed_ids = self.build_candidate_catalog(main_meal_only)
        matrix = matrix.loc[matrix.index.intersection(allowed_ids)]
        if preferred_cuisines and "cuisine_group" in self.foods_model.columns:
            allowed = {str(v).strip().lower() for v in preferred_cuisines}
            lookup = self.foods_model[["food_id", "cuisine_group"]].copy()
            lookup["food_id"] = lookup["food_id"].astype(str)
            allowed_ids = set(lookup.loc[
                lookup["cuisine_group"].astype(str).str.strip().str.lower().isin(allowed),
                "food_id",
            ])
            matrix = matrix.loc[matrix.index.intersection(allowed_ids)]

        if strategy == "average":
            group_score = matrix.mean(axis=1)
        elif strategy == "least_misery":
            group_score = matrix.min(axis=1)
        elif strategy == "most_pleasure":
            group_score = matrix.max(axis=1)
        else:
            raise ValueError("Chiến lược nhóm không hợp lệ")

        ranking = group_score[~group_score.index.isin(seen)].sort_values(ascending=False).head(top_k)
        result = ranking.rename("group_score").reset_index()
        metadata_cols = [c for c in ["food_id", "dish_name", "cuisine_group", "dish_type", "price_range"] if c in self.foods_model]
        metadata = self.foods_model[metadata_cols].copy()
        metadata["food_id"] = metadata["food_id"].astype(str)
        result = result.merge(metadata, on="food_id", how="left")
        for user_id in user_ids:
            result[f"score_{user_id}"] = result["food_id"].map(matrix[str(user_id)])
        score_cols = [f"score_{u}" for u in user_ids]
        result["minimum_score"] = result[score_cols].min(axis=1)
        result["fairness_gap"] = result[score_cols].max(axis=1) - result[score_cols].min(axis=1)
        return result, matrix

    def recommend_places_for_foods(self, food_ids: Sequence[str], top_n: int = 10) -> pd.DataFrame:
        matched = self.place_food_map[
            self.place_food_map["food_id"].astype(str).isin(set(map(str, food_ids)))
        ].copy()
        if matched.empty:
            return pd.DataFrame()
        result = (
            matched.groupby("place_id")
            .agg(
                matched_food_count=("food_id", "nunique"),
                matched_food_ids=("food_id", lambda x: ", ".join(sorted(set(map(str, x))))),
            )
            .reset_index()
            .merge(self.places, on="place_id", how="left")
        )
        result["avg_rating"] = pd.to_numeric(result.get("avg_rating", 0), errors="coerce").fillna(0)
        result["review_count"] = pd.to_numeric(result.get("review_count", 0), errors="coerce").fillna(0)
        result["place_score"] = (
            result["matched_food_count"]
            + 0.5 * result["avg_rating"]
            + 0.1 * np.log1p(result["review_count"])
        )
        cols = [c for c in [
            "place_name", "district", "matched_food_count", "avg_rating",
            "review_count", "price_range", "google_maps_link", "place_score",
        ] if c in result.columns]
        return result.sort_values("place_score", ascending=False)[cols].head(top_n)


def _safe_svd_predictions(train_ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, float]:
    matrix = train_ratings.pivot_table(index="user_id", columns="food_id", values="rating", aggfunc="mean").sort_index()
    item_mean = train_ratings.groupby("food_id")["rating"].mean()
    global_mean = float(train_ratings["rating"].mean())
    if matrix.empty:
        return pd.DataFrame(), item_mean, global_mean

    means = matrix.mean(axis=1)
    centered = matrix.sub(means, axis=0).fillna(0)
    max_components = min(centered.shape) - 1
    if max_components < 1:
        baseline = pd.DataFrame(index=centered.index, columns=centered.columns, dtype=float)
        for user_id in baseline.index:
            baseline.loc[user_id] = item_mean.reindex(baseline.columns).fillna(global_mean).to_numpy()
        return baseline, item_mean, global_mean

    n_components = min(20, max_components)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    user_factors = svd.fit_transform(centered)
    item_factors = svd.components_.T
    predictions = pd.DataFrame(
        user_factors @ item_factors.T,
        index=centered.index,
        columns=centered.columns,
    ).add(means, axis=0)
    return predictions, item_mean, global_mean


def build_system(food_file, rating_file) -> RecommendationSystem:
    foods = pd.read_excel(food_file, sheet_name="foods")
    places = pd.read_excel(food_file, sheet_name="places")
    place_food_map = pd.read_excel(food_file, sheet_name="place_food_map")
    users = pd.read_excel(rating_file, sheet_name="users")
    rated_only = pd.read_excel(rating_file, sheet_name="rated_only")

    required_food_cols = {"food_id", "dish_name"}
    required_rating_cols = {"user_id", "food_id", "rating"}
    if not required_food_cols.issubset(foods.columns):
        raise ValueError(f"Sheet foods thiếu cột: {sorted(required_food_cols - set(foods.columns))}")
    if not required_rating_cols.issubset(rated_only.columns):
        raise ValueError(f"Sheet rated_only thiếu cột: {sorted(required_rating_cols - set(rated_only.columns))}")

    foods["food_id"] = foods["food_id"].astype(str)
    valid_ids = set(foods["food_id"])
    ratings = rated_only[["user_id", "food_id", "rating"]].copy()
    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["food_id"] = ratings["food_id"].astype(str)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = (
        ratings.dropna()
        .query("1 <= rating <= 5")
        .loc[lambda x: x["food_id"].isin(valid_ids)]
        .groupby(["user_id", "food_id"], as_index=False)["rating"].mean()
    )
    if ratings.empty:
        raise ValueError("Không có rating hợp lệ sau khi làm sạch dữ liệu.")

    foods_model = foods.copy().reset_index(drop=True)
    text_cols = [c for c in ["dish_name", "description", "tags", "feature_text"] if c in foods_model]
    cat_cols = [c for c in ["cuisine_group", "dish_type", "main_ingredient", "price_range"] if c in foods_model]
    num_cols = [c for c in ["spicy_level", "healthy_score", "date_score", "popularity_score"] if c in foods_model]
    foods_model["text_feature"] = foods_model[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    for col in cat_cols:
        foods_model[col] = foods_model[col].fillna("unknown").astype(str)
    for col in num_cols:
        foods_model[col] = pd.to_numeric(foods_model[col], errors="coerce").fillna(0)

    transformers = [("text", TfidfVectorizer(max_features=2500, ngram_range=(1, 2)), "text_feature")]
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    preprocessor = ColumnTransformer(transformers=transformers)
    food_matrix = normalize(preprocessor.fit_transform(foods_model))
    food_id_to_idx = {fid: idx for idx, fid in enumerate(foods_model["food_id"].astype(str))}

    cf_predictions, item_mean, global_mean = _safe_svd_predictions(ratings)
    popularity_map = (
        foods_model.set_index(foods_model["food_id"].astype(str))["popularity_score"]
        if "popularity_score" in foods_model.columns
        else pd.Series(0.0, index=foods_model["food_id"].astype(str))
    )
    place_food_map["food_id"] = place_food_map["food_id"].astype(str)

    return RecommendationSystem(
        foods=foods,
        places=places,
        place_food_map=place_food_map,
        users=users,
        ratings=ratings,
        train_ratings=ratings,
        foods_model=foods_model,
        food_matrix=food_matrix,
        food_id_to_idx=food_id_to_idx,
        cf_predictions=cf_predictions,
        item_mean=item_mean,
        global_mean=global_mean,
        popularity_map=popularity_map,
    )


def validate_workbook(file_obj, required_sheets: set[str]) -> tuple[bool, list[str]]:
    xls = pd.ExcelFile(file_obj)
    missing = sorted(required_sheets - set(xls.sheet_names))
    return not missing, missing


def find_default_files(data_dir: str | Path = "data") -> tuple[Path | None, Path | None]:
    data_dir = Path(data_dir)
    food_file = None
    rating_file = None
    for path in list(data_dir.glob("*.xlsx")) + list(Path(".").glob("*.xlsx")):
        try:
            sheets = set(pd.ExcelFile(path).sheet_names)
        except Exception:
            continue
        if REQUIRED_FOOD_SHEETS.issubset(sheets):
            food_file = path
        if REQUIRED_RATING_SHEETS.issubset(sheets):
            rating_file = path
    return food_file, rating_file
