import glob
import functools
import streamlit as st
import os
import omegaconf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go


chart = functools.partial(st.plotly_chart, use_container_width=True)
OUTPUTS_PATH = "../../outputs"


def main() -> None:
    path = os.path.join(OUTPUTS_PATH, 'plot_curves_moco')
    datasets = [pd.read_csv(name, index_col=0) for name in glob.glob(os.path.join(path, "*all*"))]
    names = [os.path.basename(name) for name in glob.glob(os.path.join(path, "*all*"))]
    attributes = datasets[0].columns[1:].tolist()
    for attribute in attributes:
        st.markdown(f"##### {attribute}")
        fig = make_subplots(rows=1, cols=len(datasets))
        for i, dataset in enumerate(datasets):
            fig.add_trace(
                go.Scatter(x=dataset['epoch'], y=dataset[attribute], name=names[i]),
                row=1, col=i+1
            )
        fig.update_layout(height=600, width=800)
        chart(fig)
    
    for attribute in attributes:
        st.markdown(f"##### {attribute}")
        fig = make_subplots(rows=1, cols=1)
        for i, dataset in enumerate(datasets):
            fig.add_trace(
                go.Scatter(x=dataset['epoch'], y=dataset[attribute], name=names[i]),
                row=1, col=1
            )
        fig.update_layout(height=600, width=800)
        chart(fig)


if __name__ == "__main__":
    st.set_page_config(
        "Attributes: результаты",
        "📊",
        initial_sidebar_state="expanded",
        layout="wide",
    )
    main()
