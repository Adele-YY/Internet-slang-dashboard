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
    '广西': [22.8170, 108.3200], '江苏': [32.0603, 118.7969], '天津': [39.0841, 117.2008]
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

    # 1. 修正列名逻辑，避免 DuplicateError
    rename_dict = {
        '序号': 'Order', '来自IP': 'IP', '1、您的性别': 'Gender',
        '2、您的年龄段': 'Age', '3、您的常住地': 'Residence',
        '4、您是否为澳门大学在读学生': 'UM Student',
        '5、您日常刷短视频的频率': 'Frequency',
        '7、您获取网络热梗的主要渠道有哪些': 'Acquisition Channel',
        '9、您通常在哪些场景下会使用网络热梗': 'Using Scene',
        '10、您认为网络热梗对您的日常交流有何影响': 'Influence'
    }

    temp_cols = list(df.columns)
    # 给热梗评分列赋予唯一名称，而不是全部叫同一个名字
    for i in range(7):
        temp_cols[11 + i] = f"Score_Aw_{i}"
        temp_cols[19 + i] = f"Score_Us_{i}"
    df.columns = temp_cols
    df.rename(columns=rename_dict, inplace=True)

    # 2. 数据清洗
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'女': 'Female', '男': 'Male'})

    if 'Age' in df.columns:
        df['Age'] = df['Age'].replace({'18岁以下': 'Under 18', '18～30岁': '18-30', '30岁以上': 'Over 30'})
        df['Age'] = pd.Categorical(df['Age'], categories=['Under 18', '18-30', 'Over 30'], ordered=True)

    if 'UM Student' in df.columns:
        df['UM Student'] = df['UM Student'].replace({'是': 'UM Student', '否': 'Non-UM Student'})

    # 3. 评分计算
    cleanup_map = {'经常刷/听到': 2, '有印象': 1, '没听过': 0, '经常会用': 2, '有时会用': 1, '从来不用': 0}
    
    # 使用 filter 选取带有特定前缀的列
    h_data = df.filter(like="Score_Aw_").apply(lambda x: pd.to_numeric(x.replace(cleanup_map), errors='coerce').fillna(0))
    a_data = df.filter(like="Score_Us_").apply(lambda x: pd.to_numeric(x.replace(cleanup_map), errors='coerce').fillna(0))

    df['Hearing Score'] = h_data.sum(axis=1)
    df['Using Score'] = a_data.sum(axis=1)
    df['Total Score'] = df['Hearing Score'] + df['Using Score']

    # 4. 地理坐标解析
    def extract_loc(val):
        match = re.search(r'\((.*?)\)', str(val))
        return match.group(1).split('-')[0] if match else "Unknown"

    df['Location Name'] = df['IP'].apply(extract_loc)

    def get_lat_lon_with_jitter(loc):
        base = GEO_MAP.get(loc, [None, None])
        if base[0] is not None:
            # 添加随机抖动
            return base[0] + np.random.uniform(-0.15, 0.15), base[1] + np.random.uniform(-0.15, 0.15)
        return None, None

    coords = df['Location Name'].apply(get_lat_lon_with_jitter)
    df['lat'], df['lon'] = zip(*coords) if not df.empty else ([], [])

    return df, h_data, a_data

# --- 数据加载 ---
try:
    df, h_df, a_df = load_and_fully_clean_data('internet_slang.csv')
except Exception as e:
    st.error(f"Error Loading Data: {e}")
    st.stop()

# --- 侧边栏 ---
st.sidebar.header("Filters")
gen_list = st.sidebar.multiselect("Gender", df['Gender'].unique(), default=df['Gender'].unique())
age_list = st.sidebar.multiselect("Age Group", df['Age'].unique(), default=df['Age'].unique())

# 核心过滤逻辑
f_df = df[(df['Gender'].isin(gen_list)) & (df['Age'].isin(age_list))]

# --- UI 界面 ---
st.title("🌐 Internet Slang Analytics Dashboard")

# 1. 核心 KPI 与 甜甜圈图
st.divider()
kpi_col, pie_col = st.columns([3, 1])

with kpi_col:
    st.markdown("#### Key Metrics")
    m1, m2, m3, m4 = st.columns(4)
    # 增加空数据防御
    avg_h = round(f_df['Hearing Score'].mean(), 2) if not f_df.empty else 0
    avg_u = round(f_df['Using Score'].mean(), 2) if not f_df.empty else 0
    avg_t = round(f_df['Total Score'].mean(), 2) if not f_df.empty else 0
    
    m1.metric("Avg Awareness", avg_h)
    m2.metric("Avg Usage", avg_u)
    m3.metric("Avg Total", avg_t)
    m4.metric("Samples", len(f_df))

with pie_col:
    if not f_df.empty and 'UM Student' in f_df.columns:
        um_counts = f_df['UM Student'].value_counts().reset_index()
        um_counts.columns = ['Status', 'Count']
        fig_donut = px.pie(
            um_counts, values='Count', names='Status', hole=0.5,
            title="UM Student Ratio",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_donut.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_donut, use_container_width=True)

# 2. 地理分布
st.subheader("📍 Geographic Distribution")
map_data = f_df.dropna(subset=['lat'])
if not map_data.empty:
    fig_map = px.scatter_mapbox(
        map_data, lat="lat", lon="lon", color="Gender",
        hover_name="Location Name",
        hover_data={"lat": False, "lon": False, "Total Score": True},
        zoom=3.5, height=500, mapbox_style="open-street-map"
    )
    fig_map.update_traces(marker={'size': 10, 'opacity': 0.7})
    st.plotly_chart(fig_map, use_container_width=True)

# 3. 热梗综合对比
st.divider()
st.subheader("📊 Slang Landscape: Awareness vs. Usage")
if not f_df.empty:
    comp_list = []
    # 确保根据过滤后的索引提取评分
    curr_h = h_df.loc[f_df.index]
    curr_a = a_df.loc[f_df.index]
    
    for i in range(len(SLANG_CONTENT)):
        avg_aw = curr_h.iloc[:, i].mean()
        avg_us = curr_a.iloc[:, i].mean()
        comp_list.append({
            "Slang": SLANG_CONTENT[i],
            "Awareness Score": avg_aw,
            "Usage Score": avg_us,
            "Total": avg_aw + avg_us
        })
    comp_df = pd.DataFrame(comp_list)
    fig_comp = px.scatter(
        comp_df, x="Awareness Score", y="Usage Score", color="Slang",
        size="Total", text="Slang", height=500
    )
    fig_comp.update_traces(textposition='top center')
    st.plotly_chart(fig_comp, use_container_width=True)

# 4. 年龄趋势
st.divider()
st.subheader("📈 Age Trend")
if not f_df.empty:
    a_trend = f_df.groupby('Age', observed=False)[['Hearing Score', 'Using Score']].mean().reset_index()
    a_trend = a_trend.sort_values('Age')
    fig_age = px.line(a_trend, x='Age', y=['Hearing Score', 'Using Score'], markers=True)
    st.plotly_chart(fig_age, use_container_width=True)

# 5. 单项分析 (Bar Chart - 补全部分)
st.divider()
st.subheader("🔎 Individual Slang Item Analysis")
slang_idx = st.selectbox("Select Slang to Inspect:", range(
    len(SLANG_CONTENT)), format_func=lambda x: SLANG_CONTENT[x])

if not f_df.empty:
    h_col, u_col = f"Score_Aw_{slang_idx}", f"Score_Us_{slang_idx}"
    # 强制转换类型以防万一
    item_df = f_df.copy()
    item_df[h_col] = item_df[h_col].astype(float)
    item_df[u_col] = item_df[u_col].astype(float)

    item_avg = item_df.groupby('Gender')[[h_col, u_col]].mean().reset_index()
    item_avg.columns = ['Gender', 'Awareness', 'Usage']

    fig_ind = px.bar(item_avg, x='Gender', y=[
                     'Awareness', 'Usage'], barmode='group', title=f"Comparison: {SLANG_CONTENT[slang_idx]}")
    st.plotly_chart(fig_ind, use_container_width=True)
