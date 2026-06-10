from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from recommender import (
    REQUIRED_FOOD_SHEETS,
    REQUIRED_RATING_SHEETS,
    build_system,
    find_default_files,
    validate_workbook,
)

st.set_page_config(
    page_title="Hôm nay ăn gì?",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
:root{
  --bg:#FFF9F6; --surface:#FFFFFF; --surface-2:#FFF4F1;
  --rose-50:#FFF2F5; --rose-100:#FDE4EC; --rose-200:#F8C9D7;
  --rose-400:#F1A9BF; --rose-500:#E889AA; --rose-600:#C96F91;
  --ink:#3F3438; --muted:#7C6B70; --line:#F2DCE4; --success:#E7F4EC;
}
.stApp{background:linear-gradient(180deg,#FFF7F4 0,#FFFDFC 330px,#FFFFFF 100%);color:var(--ink)}
.block-container{padding-top:.75rem;padding-bottom:4rem;max-width:1220px}
h1,h2,h3,h4{color:var(--ink);letter-spacing:-.02em}

/* Hero */
.hero{position:relative;overflow:hidden;background:rgba(255,255,255,.88);border:1px solid var(--line);border-radius:28px;padding:34px 38px 30px;box-shadow:0 18px 55px rgba(216,137,170,.12);margin:4px 0 20px}
.hero:after{content:"";position:absolute;width:270px;height:270px;border-radius:50%;right:-90px;top:-130px;background:radial-gradient(circle,#FFE1E8 0,#FFF0F4 58%,transparent 60%)}
.hero-kicker{display:inline-flex;align-items:center;gap:8px;background:var(--rose-50);color:#B65F82;border:1px solid var(--rose-100);border-radius:999px;padding:7px 12px;font-size:.82rem;font-weight:700}
.hero h1{font-size:2.35rem;line-height:1.08;margin:16px 0 9px;max-width:720px}
.hero p{font-size:1.03rem;color:var(--muted);max-width:720px;margin:0}
.hero-note{margin-top:18px;color:#9A7583;font-size:.86rem}

/* Small overview stats */
.stats-strip{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 16px}
.stat-pill{background:#FFF;border:1px solid var(--line);border-radius:999px;padding:7px 12px;color:var(--muted);font-size:.82rem}
.stat-pill b{color:var(--rose-600);font-size:.93rem;margin-right:4px}

/* Navigation: make recommendation modes dominant */
[data-baseweb="tab-list"]{gap:8px;background:#FFF;border:1px solid var(--line);padding:7px;border-radius:18px;box-shadow:0 7px 24px rgba(226,153,181,.08);margin:8px 0 20px}
button[data-baseweb="tab"]{min-height:48px;padding:0 18px;border-radius:12px;font-size:.94rem;font-weight:650;color:#725F66}
button[data-baseweb="tab"]:nth-child(-n+2){min-height:58px;font-size:1.05rem;flex:1;background:#FFF7F9}
button[data-baseweb="tab"][aria-selected="true"]{background:var(--rose-100)!important;color:#A75676!important}
[data-baseweb="tab-highlight"]{display:none}

/* Forms/cards */
[data-testid="stForm"],.preference-card{background:#FFF;border:1px solid var(--line);border-radius:22px;padding:20px;box-shadow:0 9px 30px rgba(226,153,181,.08)}
.preference-card{margin-bottom:12px;padding-bottom:10px}
.quick-note{background:#FFF3F6;border:1px solid #F5D9E3;padding:12px 15px;border-radius:14px;color:#6F555E;margin-bottom:16px}
.small-note{color:var(--muted);font-size:.88rem}

/* Inputs: muted pastel, no saturated chips */
:root{color-scheme:light}
[data-baseweb="select"]>div,[data-testid="stTextInput"] input{border-radius:13px!important;background:#FFFCFB!important;border-color:#F0DEE3!important;box-shadow:none!important}
/* Force readable text on mobile browsers and dark-mode devices */
[data-baseweb="select"],
[data-baseweb="select"] div,
[data-baseweb="select"] span,
[data-baseweb="select"] input,
[data-testid="stTextInput"] input{
  color:#3F3438!important;
  -webkit-text-fill-color:#3F3438!important;
  opacity:1!important;
}
[data-baseweb="select"] input::placeholder,
[data-testid="stTextInput"] input::placeholder{
  color:#8A737B!important;
  -webkit-text-fill-color:#8A737B!important;
  opacity:1!important;
}
[data-baseweb="select"] svg{color:#9B7C87!important;fill:#9B7C87!important}
[data-baseweb="select"]>div:focus-within{border-color:#E9A9BF!important;box-shadow:0 0 0 2px #FFF0F4!important}
[data-baseweb="tag"]{background:#FFF0F4!important;color:#8A5D6D!important;border:1px solid #F4CEDA!important;border-radius:999px!important}
[data-baseweb="tag"] span{color:#8A5D6D!important}
[data-baseweb="tag"] svg{fill:#B78094!important;color:#B78094!important}
[data-baseweb="tag"]:hover{background:#FCE4EC!important;border-color:#F0BFD0!important}
[data-testid="stCheckbox"] [data-checked="true"]{background:#E889AA!important;border-color:#E889AA!important}
[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked){background:#FFF0F4;border-color:#F1B8CB}
/* Mobile/in-app browser text fix: keep labels readable without harsh black */
[data-testid="stWidgetLabel"] p,
[data-testid="stRadio"] label p,
[data-testid="stRadio"] label span,
[data-testid="stCheckbox"] label p,
[data-testid="stCheckbox"] label span,
[data-testid="stCaptionContainer"] p,
[data-testid="stMarkdownContainer"] p{
  color:#5F4A52!important;
  -webkit-text-fill-color:#5F4A52!important;
  opacity:1!important;
}
[data-testid="stRadio"] div[role="radiogroup"] label{
  color:#5F4A52!important;
  background:#FFFDFC!important;
  border:1px solid transparent!important;
  border-radius:12px!important;
  padding:8px 10px!important;
}
[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked){
  background:#FDEBF2!important;
  border-color:#F3C6D7!important;
}
/* Keep Streamlit widget labels visible in iOS/Messenger webviews */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] > div,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
.stSelectbox label,
.stMultiSelect label,
.stRadio > label,
.stSlider > label {
  display:block!important;
  visibility:visible!important;
  opacity:1!important;
  color:#4F3F46!important;
  -webkit-text-fill-color:#4F3F46!important;
  text-shadow:none!important;
}
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span {
  font-size:.95rem!important;
  font-weight:500!important;
  line-height:1.35!important;
}
@media (max-width: 768px){
  [data-testid="stWidgetLabel"],
  [data-testid="stWidgetLabel"] p,
  [data-testid="stWidgetLabel"] span {
    display:block!important;
    visibility:visible!important;
    opacity:1!important;
    color:#4F3F46!important;
    -webkit-text-fill-color:#4F3F46!important;
  }
}

/* Prevent Messenger/Safari dark mode from whitening form labels */
@media (prefers-color-scheme: dark){
  .stApp{background:linear-gradient(180deg,#FFF7F4 0,#FFFDFC 330px,#FFFFFF 100%)!important;color:#40343A!important}
  [data-testid="stWidgetLabel"] p,
  [data-testid="stRadio"] label p,
  [data-testid="stRadio"] label span,
  [data-testid="stCaptionContainer"] p{
    color:#5F4A52!important;
    -webkit-text-fill-color:#5F4A52!important;
  }
}
[data-testid="stSlider"] [role="slider"]{background:#E889AA!important;border-color:#E889AA!important}
[data-testid="stSlider"] div[data-baseweb="slider"]>div>div{background:#F6C7D8!important}

/* Buttons */
.stButton>button{border-radius:14px;min-height:44px}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#F0A4BC,#E889AA);color:white;border:0;font-weight:750;box-shadow:0 8px 18px rgba(232,137,170,.24)}
.stButton>button[kind="primary"]:hover{background:linear-gradient(135deg,#E995B1,#D9789B);color:white;transform:translateY(-1px)}

/* Results */
.result-card{background:#FFF;border:1px solid var(--line);border-radius:20px;padding:18px 20px;margin:10px 0;box-shadow:0 8px 25px rgba(226,153,181,.08)}
.result-card .score{display:inline-block;background:var(--success);color:#4E7461;border-radius:999px;padding:5px 10px;font-weight:700;font-size:.82rem}
.result-card h4{margin:10px 0 5px;font-size:1.08rem}.result-card p{margin:0;color:var(--muted);font-size:.9rem}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:16px;overflow:hidden}

/* Pastel HTML tables: avoids black canvas tables on mobile/in-app browsers */
.pastel-table-wrap{
  width:100%;overflow-x:auto;border:1px solid var(--line);border-radius:16px;
  background:#FFFFFF;box-shadow:0 8px 24px rgba(226,153,181,.07);margin:.35rem 0 1rem;
}
.pastel-table{width:100%;border-collapse:separate;border-spacing:0;min-width:720px;background:#FFFFFF;color:#4F3F46;font-size:.88rem}
.pastel-table thead th{position:sticky;top:0;background:#FDEBF2;color:#7A5262;font-weight:700;text-align:left;padding:12px 14px;border-bottom:1px solid #F2DCE4;white-space:nowrap}
.pastel-table tbody td{padding:11px 14px;border-bottom:1px solid #F7E8ED;color:#5F4A52;background:#FFFFFF;vertical-align:top}
.pastel-table tbody tr:nth-child(even) td{background:#FFFAFC}
.pastel-table tbody tr:hover td{background:#FFF2F6}
.pastel-table tbody tr:last-child td{border-bottom:0}
.pastel-table a{color:#C96F91;text-decoration:none;font-weight:650}
.pastel-table a:hover{text-decoration:underline}
@media (max-width:768px){
  .pastel-table{min-width:640px;font-size:.82rem}
  .pastel-table thead th,.pastel-table tbody td{padding:10px 12px}
}

/* Hide Streamlit chrome noise */
#MainMenu{visibility:hidden} footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Đang xây dựng mô hình gợi ý...")
def load_system(food_source, rating_source):
    return build_system(food_source, rating_source)


def pct(series: pd.Series) -> pd.Series:
    return (pd.to_numeric(series, errors="coerce").fillna(0) * 100).round(1)


def render_pastel_table(df: pd.DataFrame, max_rows: int | None = None) -> None:
    """Render a light pastel table that stays readable on mobile webviews."""
    table = df.copy()
    if max_rows is not None:
        table = table.head(max_rows)

    # Make map URLs compact and clickable instead of printing long links.
    def linkify(value):
        text = str(value)
        if text.startswith(("http://", "https://")):
            return f'<a href="{text}" target="_blank" rel="noopener noreferrer">Mở bản đồ</a>'
        return text

    for col in table.columns:
        if "link" in str(col).lower() or "url" in str(col).lower():
            table[col] = table[col].map(linkify)

    html = table.to_html(index=False, escape=False, classes="pastel-table", border=0)
    st.markdown(f'<div class="pastel-table-wrap">{html}</div>', unsafe_allow_html=True)


def format_food_table(df: pd.DataFrame, group: bool = False) -> pd.DataFrame:
    out = df.copy()
    rename = {
        "food_id": "Mã món",
        "dish_name": "Tên món",
        "cuisine_group": "Ẩm thực",
        "dish_type": "Loại món",
        "main_ingredient": "Nguyên liệu chính",
        "price_range": "Khoảng giá",
        "spicy_level": "Độ cay",
        "healthy_score": "Điểm lành mạnh",
        "hybrid_score": "Độ phù hợp (%)",
        "group_score": "Điểm nhóm (%)",
        "minimum_score": "Điểm thấp nhất (%)",
        "fairness_gap": "Chênh lệch sở thích (%)",
    }
    if "hybrid_score" in out:
        out["hybrid_score"] = pct(out["hybrid_score"])
    if group:
        for col in [c for c in out.columns if c.startswith("score_")]:
            out[col] = pct(out[col])
            rename[col] = f"Điểm {col.removeprefix('score_')} (%)"
        for col in ["group_score", "minimum_score", "fairness_gap"]:
            if col in out:
                out[col] = pct(out[col])
    keep = [c for c in [
        "food_id", "dish_name", "cuisine_group", "dish_type", "main_ingredient",
        "price_range", "spicy_level", "healthy_score", "hybrid_score",
    ] if c in out.columns]
    if group:
        keep = [c for c in [
            "food_id", "dish_name", "cuisine_group", "dish_type", "price_range",
            *[x for x in out.columns if x.startswith("score_")],
            "minimum_score", "fairness_gap", "group_score",
        ] if c in out.columns]
    return out[keep].rename(columns=rename)


st.markdown("""
<div class="hero">
  <span class="hero-kicker">🍜 HANOI FOOD MATCH</span>
  <h1>Hôm nay ăn gì?</h1>
  <p>Tìm món phù hợp cho riêng bạn hoặc cân bằng sở thích của hai người — theo khẩu vị, ngân sách và khu vực.</p>
  <div class="hero-note">Chọn vài sở thích cơ bản, hệ thống sẽ gợi ý món ăn và quán phù hợp cho bạn.</div>
</div>
""", unsafe_allow_html=True)

default_food, default_rating = find_default_files("data")

with st.sidebar:
    st.header("Dữ liệu")
    if default_food and default_rating:
        st.success("Đã tìm thấy dữ liệu trong thư mục data/")
        use_upload = st.toggle("Dùng file tải lên thay thế", value=False)
    else:
        use_upload = True
        st.info("Tải lên hai file Excel để chạy demo.")

    food_upload = rating_upload = None
    if use_upload:
        food_upload = st.file_uploader(
            "Food master (.xlsx)", type=["xlsx"],
            help="Cần các sheet: foods, places, place_food_map",
        )
        rating_upload = st.file_uploader(
            "Dữ liệu rating (.xlsx)", type=["xlsx"],
            help="Cần các sheet: users, user_ratings, rated_only",
        )

if use_upload:
    if not food_upload or not rating_upload:
        st.info("Hãy tải đủ hai file Excel ở thanh bên trái để bắt đầu.")
        st.stop()
    try:
        ok_food, missing_food = validate_workbook(food_upload, REQUIRED_FOOD_SHEETS)
        food_upload.seek(0)
        ok_rating, missing_rating = validate_workbook(rating_upload, REQUIRED_RATING_SHEETS)
        rating_upload.seek(0)
        if not ok_food:
            st.error(f"Food master thiếu sheet: {', '.join(missing_food)}")
            st.stop()
        if not ok_rating:
            st.error(f"File rating thiếu sheet: {', '.join(missing_rating)}")
            st.stop()
        system = load_system(food_upload, rating_upload)
    except Exception as exc:
        st.exception(exc)
        st.stop()
else:
    try:
        system = load_system(str(default_food), str(default_rating))
    except Exception as exc:
        st.exception(exc)
        st.stop()

place_count = system.places["place_id"].nunique() if "place_id" in system.places else len(system.places)
st.markdown(
    f"""
    <div class="stats-strip">
        <span class="stat-pill"><b>{system.foods['food_id'].nunique():,}</b> món ăn</span>
        <span class="stat-pill"><b>{place_count:,}</b> quán</span>
        <span class="stat-pill"><b>{system.ratings['user_id'].nunique():,}</b> người khảo sát</span>
        <span class="stat-pill"><b>{len(system.ratings):,}</b> lượt đánh giá</span>
    </div>
    """,
    unsafe_allow_html=True,
)

personal_tab, group_tab, explore_tab, model_tab = st.tabs([
    "Gợi ý cá nhân", "Gợi ý cho hai người", "Khám phá món", "Về hệ thống",
])

with personal_tab:
    st.subheader("Tìm món phù hợp với sở thích của bạn")
    personal_mode = st.radio(
        "Bạn muốn nhận gợi ý theo cách nào?",
        ["Tôi là người dùng mới", "Tôi đã có lịch sử đánh giá"],
        horizontal=True,
        key="personal_mode",
    )

    if personal_mode == "Tôi là người dùng mới":
        st.caption("Chọn nhanh một vài tiêu chí. Nếu để trống, hệ thống sẽ gợi ý các món phù hợp chung trong khảo sát.")

        cuisine_vi = {
            "Vietnamese": "Việt Nam", "Chinese/Taiwanese": "Trung Hoa/Đài Loan",
            "Japanese": "Nhật Bản", "Korean": "Hàn Quốc", "Thai": "Thái Lan",
            "Western/European": "Âu/Mỹ", "Indian": "Ấn Độ", "Indonesian": "Indonesia",
        }
        cuisine_display = {cuisine_vi.get(x, x): x for x in system.available_cuisines}

        type_vi = {
            "noodle_dish": "Bún, phở, mì", "noodle_soup": "Món nước", "rice_dish": "Cơm",
            "hotpot": "Lẩu", "grilled_dish": "Món nướng", "fried_dish": "Món chiên",
            "roll": "Món cuốn", "seafood": "Hải sản", "snack": "Ăn nhẹ",
            "dessert": "Tráng miệng", "salad": "Rau và salad", "soup": "Canh, súp",
            "main_dish": "Món chính", "street_food": "Đồ ăn đường phố",
        }
        raw_types = sorted(system.foods_model.get("dish_type", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        type_display = {type_vi.get(x, x.replace("_", " ").title()): x for x in raw_types}

        price_vi = {"low": "Tiết kiệm", "medium": "Vừa phải", "high": "Cao cấp"}
        raw_prices = sorted(system.foods_model.get("price_range", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        price_display = {price_vi.get(x.lower(), x): x for x in raw_prices}

        raw_spicy = sorted(system.foods_model.get("spicy_level", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        spicy_vi = {"none": "Không cay", "low": "Ít cay", "medium": "Cay vừa", "high": "Cay nhiều", "mild": "Ít cay", "spicy": "Cay"}
        spicy_display = {spicy_vi.get(x.lower(), x): x for x in raw_spicy}

        avoid_map = {
            "Hải sản": ["hải sản", "seafood", "tôm", "cua", "mực", "cá"],
            "Thịt bò": ["bò", "beef"], "Thịt heo": ["heo", "lợn", "pork"],
            "Nội tạng": ["lòng", "nội tạng", "offal"], "Đồ sống": ["sashimi", "raw"],
            "Đồ cay": ["cay", "spicy"], "Đồ chiên": ["chiên", "rán", "fried"],
        }

        c1, c2 = st.columns(2, gap="large")
        with c1:
            cuisines_label = st.multiselect(
                "Bạn thích ẩm thực nào? (tối đa 3)", list(cuisine_display),
                max_selections=3, key="personal_new_cuisines",
                placeholder="Không chọn = hệ thống tự gợi ý",
            )
            types_label = st.multiselect(
                "Bạn muốn ăn kiểu món nào? (tối đa 3)", list(type_display),
                max_selections=3, key="personal_new_types",
                placeholder="Ví dụ: Lẩu, Bún/phở/mì",
            )
        with c2:
            price_label = st.multiselect(
                "Mức giá", list(price_display), max_selections=2,
                key="personal_new_prices", placeholder="Không quan trọng",
            )
            spicy_label = st.multiselect(
                "Độ cay", list(spicy_display), max_selections=2,
                key="personal_new_spicy", placeholder="Linh hoạt",
            )

        avoid_label = st.multiselect(
            "Bạn muốn tránh gì?", list(avoid_map), key="personal_new_avoid",
            placeholder="Có thể bỏ qua nếu không kiêng",
        )
        p1, p2 = st.columns([2, 1])
        main_only = p1.checkbox("Chỉ lấy món chính", value=True, key="personal_new_main")
        top_k = p2.slider("Số món", 5, 20, 10, key="personal_new_topk")

        no_preferences = not any([cuisines_label, types_label, price_label, spicy_label, avoid_label])
        if no_preferences:
            st.info("Bạn chưa chọn sở thích. Hệ thống sẽ ưu tiên các món được 59 người khảo sát đánh giá cao.")

        if st.button("✨ Gợi ý nhanh cho tôi", type="primary", use_container_width=True, key="personal_new_submit"):
            avoid_words = [word for label in avoid_label for word in avoid_map[label]]
            result = system.recommend_new_user(
                preferred_cuisines=[cuisine_display[x] for x in cuisines_label],
                preferred_dish_types=[type_display[x] for x in types_label],
                preferred_prices=[price_display[x] for x in price_label],
                preferred_spicy_levels=[spicy_display[x] for x in spicy_label],
                excluded_keywords=avoid_words,
                top_k=top_k,
                main_meal_only=main_only,
            )
            st.session_state["personal_result"] = result

    else:
        st.caption("Dành cho người dùng đã có lịch sử đánh giá trong dữ liệu thử nghiệm.")
        c1, c2, c3 = st.columns([1, 1.4, 1])
        user_id = c1.selectbox("Người dùng", system.eligible_users, key="personal_user")
        cuisines = c2.multiselect("Nhóm ẩm thực", system.available_cuisines, key="personal_cuisine")
        top_k = c3.slider("Số món", 5, 20, 10, key="personal_topk")
        o1, o2 = st.columns(2)
        main_only = o1.checkbox("Chỉ lấy món chính", value=True, key="personal_old_main")
        exclude_seen = o2.checkbox("Loại món đã đánh giá", value=True, key="personal_old_seen")

        if st.button("Gợi ý món ăn", type="primary", use_container_width=True, key="personal_old_submit"):
            result = system.recommend_user(
                user_id=user_id,
                top_k=top_k,
                preferred_cuisines=cuisines,
                main_meal_only=main_only,
                exclude_seen=exclude_seen,
            )
            st.session_state["personal_result"] = result

    result = st.session_state.get("personal_result")
    if result is not None:
        if result.empty:
            st.warning("Không tìm thấy kết quả phù hợp. Hãy nới bớt bộ lọc.")
        else:
            st.markdown("### Gợi ý nổi bật")
            for _, row in result.head(3).iterrows():
                score = float(row.get("hybrid_score", 0)) * 100
                meta = " · ".join(str(row.get(c, "")) for c in ["cuisine_group", "dish_type", "price_range"] if pd.notna(row.get(c)) and str(row.get(c)).strip())
                st.markdown(f"""<div class="result-card"><span class="score">Phù hợp {score:.0f}%</span><h4>{row.get('dish_name','Món ăn')}</h4><p>{meta}</p></div>""", unsafe_allow_html=True)
            with st.expander("Xem bảng kết quả chi tiết"):
                render_pastel_table(format_food_table(result))
            st.subheader("Quán phù hợp với các món trên")
            places = system.recommend_places_for_foods(result["food_id"].tolist(), top_n=10)
            if places.empty:
                st.info("Chưa tìm thấy quán được ánh xạ với các món này.")
            else:
                render_pastel_table(places)

with group_tab:
    st.subheader("Hai sở thích, một lựa chọn chung")
    st.caption("Chưa có lịch sử chỉ cần trả lời nhanh vài câu hỏi — không phải tìm từng món.")
    group_mode = st.radio(
        "Đối tượng sử dụng",
        ["Chưa có lịch sử", "Đã từng đánh giá"],
        horizontal=True,
        help="Chưa có lịch sử nhập sở thích trực tiếp; không cần có user_id trong dữ liệu train.",
    )

    strategy_label = st.selectbox(
        "Chiến lược ghép sở thích",
        ["Cân bằng sở thích", "Tránh món một người không thích", "Ưu tiên món có người rất thích"],
        key="group_strategy",
    )
    strategy_map = {
        "Cân bằng sở thích": "average",
        "Tránh món một người không thích": "least_misery",
        "Ưu tiên món có người rất thích": "most_pleasure",
    }

    if group_mode == "Chưa có lịch sử":
        # Human-friendly labels for fast cold-start onboarding.
        cuisine_vi = {
            "Vietnamese": "Việt Nam", "Chinese/Taiwanese": "Trung Hoa/Đài Loan",
            "Japanese": "Nhật Bản", "Korean": "Hàn Quốc", "Thai": "Thái Lan",
            "Western/European": "Âu/Mỹ", "Indian": "Ấn Độ", "Indonesian": "Indonesia",
        }
        cuisine_display = {cuisine_vi.get(x, x): x for x in system.available_cuisines}

        type_vi = {
            "noodle_dish": "Bún, phở, mì", "noodle_soup": "Món nước", "rice_dish": "Cơm",
            "hotpot": "Lẩu", "grilled_dish": "Món nướng", "fried_dish": "Món chiên",
            "roll": "Món cuốn", "seafood": "Hải sản", "snack": "Ăn nhẹ",
            "dessert": "Tráng miệng", "salad": "Rau và salad", "soup": "Canh, súp",
            "main_dish": "Món chính", "street_food": "Đồ ăn đường phố",
        }
        raw_types = sorted(system.foods_model.get("dish_type", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        type_display = {type_vi.get(x, x.replace("_", " ").title()): x for x in raw_types}

        price_vi = {"low": "Tiết kiệm", "medium": "Vừa phải", "high": "Cao cấp"}
        raw_prices = sorted(system.foods_model.get("price_range", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        price_display = {price_vi.get(x.lower(), x): x for x in raw_prices}

        raw_spicy = sorted(system.foods_model.get("spicy_level", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        spicy_vi = {"none": "Không cay", "low": "Ít cay", "medium": "Cay vừa", "high": "Cay nhiều", "mild": "Ít cay", "spicy": "Cay"}
        spicy_display = {spicy_vi.get(x.lower(), x): x for x in raw_spicy}

        avoid_map = {
            "Hải sản": ["hải sản", "seafood", "tôm", "cua", "mực", "cá"],
            "Thịt bò": ["bò", "beef"], "Thịt heo": ["heo", "lợn", "pork"],
            "Nội tạng": ["lòng", "nội tạng", "offal"], "Đồ sống": ["sashimi", "raw"],
            "Đồ cay": ["cay", "spicy"], "Đồ chiên": ["chiên", "rán", "fried"],
        }

        st.markdown('<div class="quick-note">Chỉ cần chọn vài ô nổi bật nhất. Mỗi người mất khoảng 20–30 giây, không cần tìm từng món cụ thể.</div>', unsafe_allow_html=True)

        def quick_profile(person_key: str, title: str):
            st.markdown(f'<div class="preference-card"><h4>{title}</h4>', unsafe_allow_html=True)
            cuisines_label = st.multiselect(
                "Bạn thích ẩm thực nào? (tối đa 3)", list(cuisine_display),
                max_selections=3, key=f"{person_key}_cuisines",
                placeholder="Ví dụ: Việt Nam, Hàn Quốc",
            )
            types_label = st.multiselect(
                "Bạn thường muốn ăn kiểu món nào? (tối đa 3)", list(type_display),
                max_selections=3, key=f"{person_key}_types",
                placeholder="Ví dụ: Lẩu, Món nướng",
            )
            c1, c2 = st.columns(2)
            price_label = c1.multiselect(
                "Mức giá", list(price_display), max_selections=2, key=f"{person_key}_prices",
                placeholder="Không chọn = không quan trọng",
            )
            spicy_label = c2.multiselect(
                "Độ cay", list(spicy_display), max_selections=2, key=f"{person_key}_spicy",
                placeholder="Không chọn = linh hoạt",
            )
            avoid_label = st.multiselect(
                "Bạn muốn tránh gì?", list(avoid_map), key=f"{person_key}_avoid",
                placeholder="Có thể bỏ qua nếu không kiêng",
            )
            st.markdown('</div>', unsafe_allow_html=True)
            avoid_words = [word for label in avoid_label for word in avoid_map[label]]
            return {
                "liked_food_ids": [], "disliked_food_ids": [],
                "preferred_cuisines": [cuisine_display[x] for x in cuisines_label],
                "preferred_dish_types": [type_display[x] for x in types_label],
                "preferred_prices": [price_display[x] for x in price_label],
                "preferred_spicy_levels": [spicy_display[x] for x in spicy_label],
                "excluded_keywords": avoid_words,
            }

        left, right = st.columns(2, gap="large")
        with left:
            profile_a = quick_profile("new_a", "👤 Người thứ nhất")
        with right:
            profile_b = quick_profile("new_b", "👤 Người thứ hai")

        group_top_k = st.slider("Số món muốn nhận", 5, 20, 10, key="new_group_topk")
        enough_info = any([profile_a["preferred_cuisines"], profile_a["preferred_dish_types"], profile_a["preferred_prices"], profile_a["preferred_spicy_levels"]]) and any([profile_b["preferred_cuisines"], profile_b["preferred_dish_types"], profile_b["preferred_prices"], profile_b["preferred_spicy_levels"]])
        if not enough_info:
            st.info("Mỗi người chỉ cần chọn ít nhất một sở thích, chẳng hạn nhóm ẩm thực hoặc kiểu món.")

        if st.button(
            "💗 Tìm món hợp với cả hai", type="primary", use_container_width=True, disabled=not enough_info
        ):
            result = system.group_recommend_new_users(
                [profile_a, profile_b], strategy=strategy_map[strategy_label],
                top_k=group_top_k, main_meal_only=True,
            )
            st.session_state["group_result"] = result

    else:
        users = system.eligible_users
        g1, g2 = st.columns(2)
        user_a = g1.selectbox("Người thứ nhất", users, index=0, key="group_a")
        default_b = 1 if len(users) > 1 else 0
        user_b = g2.selectbox("Người thứ hai", users, index=default_b, key="group_b")
        gc1, gc2 = st.columns([2, 1])
        group_cuisines = gc1.multiselect("Nhóm ẩm thực", system.available_cuisines, key="group_cuisine")
        group_top_k = gc2.slider("Số món", 5, 20, 10, key="group_topk")

        if user_a == user_b:
            st.warning("Hãy chọn hai người dùng khác nhau.")
        elif st.button("Tạo gợi ý cho nhóm", type="primary", use_container_width=True):
            result, _ = system.group_recommend(
                [user_a, user_b],
                strategy=strategy_map[strategy_label],
                top_k=group_top_k,
                preferred_cuisines=group_cuisines,
                main_meal_only=True,
            )
            st.session_state["group_result"] = result

    group_result = st.session_state.get("group_result")
    if group_result is not None:
        if group_result.empty:
            st.warning("Không tìm thấy món chung phù hợp với bộ lọc hiện tại.")
        else:
            st.markdown("### Những món hợp với cả hai")
            for _, row in group_result.head(3).iterrows():
                score = float(row.get("group_score", 0)) * 100
                gap = float(row.get("fairness_gap", 0)) * 100
                meta = " · ".join(str(row.get(c, "")) for c in ["cuisine_group", "dish_type", "price_range"] if pd.notna(row.get(c)) and str(row.get(c)).strip())
                st.markdown(f"""<div class="result-card"><span class="score">Hợp nhóm {score:.0f}%</span><h4>{row.get('dish_name','Món ăn')}</h4><p>{meta} · Chênh lệch sở thích {gap:.0f}%</p></div>""", unsafe_allow_html=True)
            with st.expander("Xem bảng điểm chi tiết"):
                render_pastel_table(format_food_table(group_result, group=True))
            st.caption("Chênh lệch sở thích càng thấp thì kết quả càng công bằng giữa hai người.")
            places = system.recommend_places_for_foods(group_result["food_id"].tolist(), top_n=10)
            if not places.empty:
                st.subheader("Quán phù hợp")
                render_pastel_table(places)

with explore_tab:
    st.subheader("Khám phá danh mục món ăn")
    query = st.text_input("Tìm theo tên món")
    cuisine_filter = st.multiselect("Lọc theo cuisine", system.available_cuisines, key="explore_cuisine")
    data = system.foods.copy()
    if query:
        data = data[data["dish_name"].astype(str).str.contains(query, case=False, na=False)]
    if cuisine_filter and "cuisine_group" in data:
        data = data[data["cuisine_group"].isin(cuisine_filter)]
    show_cols = [c for c in ["food_id", "dish_name", "cuisine_group", "dish_type", "main_ingredient", "price_range"] if c in data]
    render_pastel_table(data[show_cols], max_rows=500)

with model_tab:
    st.subheader("Cách hệ thống tạo gợi ý")
    st.markdown("""
- **Collaborative Filtering:** học xu hướng đánh giá giữa người dùng và món ăn bằng Truncated SVD.
- **Content-Based Filtering:** so khớp hồ sơ người dùng với tên món, mô tả, tag, cuisine, loại món và các đặc trưng số.
- **Popularity:** bổ sung tín hiệu độ phổ biến để giảm kết quả quá hẹp.
- **Hybrid score:** `0.50 × CF + 0.35 × Content + 0.15 × Popularity`.
- **Group Recommendation:** hỗ trợ trung bình, least misery và most pleasure.
""")
    st.info("Bản demo huấn luyện mô hình trực tiếp từ file dữ liệu khi ứng dụng khởi động và dùng cache để tránh chạy lại không cần thiết.")
