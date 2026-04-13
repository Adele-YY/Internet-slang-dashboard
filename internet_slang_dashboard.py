import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. 页面基础配置
st.set_page_config(page_title="Internet Slang Dashboard", layout="wide")

# 地理坐标映射
GEO_MAP = {
    '澳门': [22.1987, 113.5439], '广东': [23.1291, 113.2644], '江西': [28.6761, 115.8921],
    '北京': [39.9042, 116.4074], '河北': [38.0428, 114.5149], '四川': [30.5728, 104.0668],
    '重庆': [29.5630, 106.5516], '浙江': [30.2741, 120.1551], '上海': [31.2304, 121.4737],
    '山东': [36.6512, 117.0483], '山西': [37.8706, 112.5489], '黑龙江': [45.7569, 126.6424],
    '湖北': [30.5928, 114.3055], '安徽': [31.8612, 117.2830], '辽宁': [41.8057, 123.4315],
    '广西': [22.8170, 108.3200], '江苏': [32.0603, 118.7969], '天津': [39.0841, 117.2008],
    '国外': [20.0, 0.0]
}

# 热梗内容映射
SLANG_CONTENT = [
    "wc / 操 / 我靠",
    "老己 / 爱你老己",
    "我的身材很曼妙",
    "你 / 我已急哭",
    "洗衣粉儿",
    "心理委员我不得劲儿",
    "我不行了"
]


@st.cache_data
def load_and_fully_clean_data(file_path):
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()

    rename_dict = {
        '序号': 'Order', '来自IP': 'IP', '1、您的性别': 'Gender',
        '2、您的年龄段': 'Age', '3、您的常住地': 'Residence',
        '4、您是否为澳门大学在读学生': 'UM Student',  # 修改为更清晰的英文名
        '5、您日常刷短视频的频率': 'Frequency',
        '7、您获取网络热梗的主要渠道有哪些': 'Acquisition Channel',
        '9、您通常在哪些场景下会使用网络热梗': 'Using Scene',
        '10、您认为网络热梗对您的日常交流有何影响': 'Influence'
    }

    cols = list(df.columns)
    for i in range(7):
        cols[11 + i] = "Awareness"
        cols[19 + i] = "Usage"
    df.columns = cols
    df.rename(columns=rename_dict, inplace=True)

    # 内容清洗
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'女': 'Female', '男': 'Male'})

    if 'Age' in df.columns:
        df['Age'] = df['Age'].replace(
            {'18岁以下': 'Under 18', '18～30岁': '18-30', '30岁以上': 'Over 30'})
        age_order = ['Under 18', '18-30', 'Over 30']
        df['Age'] = pd.Categorical(
            df['Age'], categories=age_order, ordered=True)

    # 清洗澳门大学学生字段
    if 'UM Student' in df.columns:
        df['UM Student'] = df['UM Student'].replace(
            {'是': 'UM Student', '否': 'Non-UM Student'})

    cleanup_map = {'经常刷/听到': 2, '有印象': 1,
                   '没听过': 0, '经常会用': 2, '有时会用': 1, '从来不用': 0}
    h_data = df.iloc[:, 11:18].apply(lambda x: pd.to_numeric(
        x.replace(cleanup_map), errors='coerce').fillna(0))
    a_data = df.iloc[:, 19:26].apply(lambda x: pd.to_numeric(
        x.replace(cleanup_map), errors='coerce').fillna(0))

    df['Hearing Score'] = h_data.sum(axis=1)
    df['Using Score'] = a_data.sum(axis=1)
    df['Total Score'] = df['Hearing Score'] + df['Using Score']

    def extract_loc(val):
        match = re.search(r'\((.*?)\)', str(val))
        return match.group(1).split('-')[0] if match else "Unknown"

    df['Location Name'] = df['IP'].apply(extract_loc)

    def get_lat_lon_with_jitter(loc):
        base = GEO_MAP.get(loc, [None, None])
        if base[0] is not None:
            jitter_lat = base[0] + np.random.uniform(-0.15, 0.15)
            jitter_lon = base[1] + np.random.uniform(-0.15, 0.15)
            return jitter_lat, jitter_lon
        return None, None

    coords = df['Location Name'].apply(get_lat_lon_with_jitter)
    df['lat'] = [c[0] for c in coords]
    df['lon'] = [c[1] for c in coords]

    return df, h_data, a_data


# --- 数据加载 ---
try:
    df, h_df, a_df = load_and_fully_clean_data('internet_slang.csv')
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- Streamlit 界面 ---
st.title("🌐 Internet Slang Analytics Dashboard")

# 侧边栏
st.sidebar.header("Filters")
gen_list = st.sidebar.multiselect(
    "Gender", df['Gender'].unique(), default=df['Gender'].unique())
age_list = st.sidebar.multiselect(
    "Age Group", df['Age'].unique(), default=df['Age'].unique())
f_df = df[(df['Gender'].isin(gen_list)) & (df['Age'].isin(age_list))]

# 1. 核心 KPI 布局调整：加入甜甜圈图
st.divider()
kpi_col, pie_col = st.columns([3, 1])  # KPI占3份，甜甜圈图占1份

with kpi_col:
    st.markdown("#### Key Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Awareness", round(f_df['Hearing Score'].mean(), 2))
    m2.metric("Avg Usage", round(f_df['Using Score'].mean(), 2))
    m3.metric("Avg Total", round(f_df['Total Score'].mean(), 2))
    m4.metric("Samples", len(f_df))

with pie_col:
    # 绘制甜甜圈图
    if 'UM Student' in f_df.columns:
        um_data = f_df['UM Student'].value_counts().reset_index()
        um_data.columns = ['Status', 'Count']

        fig_donut = px.pie(
            um_data,
            values='Count',
            names='Status',
            hole=0.5,  # 设置中间的孔洞大小，变成甜甜圈
            title="UM Student Ratio",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_donut.update_layout(
            showlegend=True, height=300, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_donut, use_container_width=True)

# 2. 地理分布
st.subheader("📍 Data Resource")
st.markdown("#### Geographic Distribution")
map_data = f_df.dropna(subset=['lat'])
if not map_data.empty:
    fig_map = px.scatter_mapbox(
        map_data,
        lat="lat",
        lon="lon",
        color="Gender",
        hover_name="Location Name",
        hover_data={"lat": False, "lon": False, "Total Score": True},
        zoom=3.5,
        height=500,
        mapbox_style="open-street-map"
    )
    fig_map.update_traces(marker={'size': 10, 'opacity': 0.7})
    st.plotly_chart(fig_map, use_container_width=True)

# 3. Slang 综合对比图
st.divider()
st.subheader("📊 Slang Landscape: Awareness vs. Usage")
# ... (此部分代码保持不变)
comp_data = []
for i in range(len(SLANG_CONTENT)):
    avg_aw = h_df.iloc[f_df.index, i].mean()
    avg_us = a_df.iloc[f_df.index, i].mean()
    comp_data.append({
        "Slang Content": SLANG_CONTENT[i],
        "Average Awareness": avg_aw,
        "Average Usage": avg_us,
        "Engagement": avg_aw + avg_us
    })
comp_df = pd.DataFrame(comp_data)

fig_comp = px.scatter(
    comp_df, x="Average Awareness", y="Average Usage",
    color="Slang Content", size="Engagement",
    text="Slang Content",
    labels={"Average Awareness": "Awareness Score",
            "Average Usage": "Usage Score"},
    height=500
)
fig_comp.update_traces(textposition='top center')
st.plotly_chart(fig_comp, use_container_width=True)

# 4. 群体统计
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.subheader("Gender: Awareness vs Usage")
    g_data = f_df.groupby('Gender')[
        ['Hearing Score', 'Using Score']].mean().reset_index()
    g_data.rename(columns={'Hearing Score': 'Awareness Score',
                  'Using Score': 'Usage Score'}, inplace=True)
    st.plotly_chart(px.bar(g_data, x='Gender', y=[
                    'Awareness Score', 'Usage Score'], barmode='group'), use_container_width=True)

with c2:
    st.subheader("Age Trend (Sorted)")
    a_data = f_df.groupby('Age', observed=False)[
        ['Hearing Score', 'Using Score']].mean().reset_index()
    a_data.rename(columns={'Hearing Score': 'Awareness Score',
                  'Using Score': 'Usage Score'}, inplace=True)
    a_data = a_data.sort_values('Age')
    st.plotly_chart(px.line(a_data, x='Age', y=[
                    'Awareness Score', 'Usage Score'], markers=True), use_container_width=True)

# 5. 单项深入探究
st.divider()
st.subheader("🔎 Individual Slang Item Analysis")
slang_idx = st.selectbox("Select Slang to Inspect:", range(
    len(SLANG_CONTENT)), format_func=lambda x: SLANG_CONTENT[x])
item_analysis = pd.DataFrame({
    'Gender': df['Gender'],
    'Awareness': h_df.iloc[:, slang_idx],
    'Usage': a_df.iloc[:, slang_idx]
})
item_avg = item_analysis[item_analysis['Gender'].isin(
    gen_list)].groupby('Gender').mean().reset_index()
st.plotly_chart(px.bar(item_avg, x='Gender', y=['Awareness', 'Usage'], barmode='group',
                       title=f"Detailed View: {SLANG_CONTENT[slang_idx]}"), use_container_width=True)
