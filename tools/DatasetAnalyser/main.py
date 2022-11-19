from typing import List
import functools
import omegaconf
import streamlit as st
import pandas as pd
import plotly.express as px
import pydot
import cv2
import glob

############
mapping_version = "att_3"
part = "train"
base_path = omegaconf.OmegaConf.load('configs/config.yaml')["path"]
mapping = omegaconf.OmegaConf.load(f'configs/mapping/{mapping_version}.yaml')
#dataset_dependencies = omegaconf.OmegaConf.load("configs/dataset_dependencies.yaml")
#dataset_name = list(dataset_dependencies.keys())[0]
############

chart = functools.partial(st.plotly_chart, use_container_width=True)


@st.experimental_memo
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Filepath"] = df["Filepath"].apply(lambda x: x.split("/")[-3])
    attributes = [attribute.name for attribute in mapping]
    columns = ["Filepath"] + attributes
    return df[columns].rename(columns={"Filepath": "datasets"})


@st.experimental_memo
def filter_data(
    df: pd.DataFrame, dataset_selections: List[str], attribute_selections: List[str]
) -> pd.DataFrame:
    df = df.copy()
    df = df[df.datasets.isin(dataset_selections)]
    columns = ["datasets"] + attribute_selections
    return df[columns]


def build_dependencies_graph():
    graph = pydot.Dot(graph_type='digraph')
    # Convert to dict format
    dict_dep = omegaconf.OmegaConf.to_container(dataset_dependencies)

    def dfs(vertex: dict or str, parent: pydot.Node) -> None:
        if type(vertex) == list:
            for v in vertex:
                node = pydot.Node(parent.get_name() + v, label=v)
                graph.add_node(node)
                graph.add_edge(pydot.Edge(parent, node))
            return
        for name, child in vertex.items():
            node = pydot.Node(name, label=name)
            graph.add_node(node)
            graph.add_edge(pydot.Edge(parent, node))
            dfs(child, node)

    root_name = list(dict_dep.keys())[0]
    root = pydot.Node(root_name)
    dfs(dict_dep[root_name], root)
    graph.write_png('generated/dependencies_tree.png')


def main() -> None:
    st.header(f"Attributes")
    csv_path = st.selectbox(
        'Выберите набор',
        glob.glob(f'{base_path}/*.csv'))


#    build_dependencies_graph()
#
#    with st.expander("Иерархия"):
#        image = cv2.imread('generated/dependencies_tree.png')
#        st.image(image)

    df = pd.read_csv(csv_path)
    df = clean_data(df)

    st.sidebar.subheader("Фильтрация по наборам")
    datasets = list(df.datasets.unique())
    dataset_selections = st.sidebar.multiselect(
        "Укажите наборы данных", options=datasets, default=datasets
    )

    st.sidebar.subheader("Фильтрация по атрибутам")

    attributes = [attribute.name for attribute in mapping]
    attribute_selections = st.sidebar.multiselect(
        "Укажите атрибуты", options=attributes, default=attributes
    )

    st.subheader(f"Всего картинок: {len(df.index)}")

    st.subheader("Соотношение размеров наборов")
    pie = df.groupby(["datasets"])["datasets"].count().to_frame()
    pie["name"] = pie.index
    pie = pie.rename(columns={"datasets": "images count"})
    fig = px.pie(pie, values="images count", names="name")
    chart(fig)

    df = filter_data(df, dataset_selections, attribute_selections)
    for id, attribute in enumerate(attributes):
        st.subheader(f"{attribute}")
        count = df.groupby([attribute])[attribute].count().to_frame()
        count["value"] = count.index
        count["value"] = count["value"].apply(lambda x: mapping[id]["values"][x] if x != "-1" else "Непонятно")
        fig = px.bar(count, y=attribute, x="value", color='value')
        fig.update_layout(barmode="stack", xaxis={"categoryorder": "total descending"})
        fig.update(layout_coloraxis_showscale=False)
        chart(fig)


if __name__ == "__main__":
    st.set_page_config(
        "Attributes: наборы данных",
        "📊",
        initial_sidebar_state="expanded",
        layout="wide",
    )
    main()
