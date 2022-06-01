import platform
from typing import Tuple, List

import streamlit as st
import torch
from trecover.config import var
from trecover.utils.beam_search import beam_search, dashboard_loop
from trecover.utils.inference import data_to_columns, create_noisy_columns
from trecover.utils.transform import columns_to_tensor, tensor_to_target
from trecover.utils.visualization import visualize_columns, visualize_target

max_chars = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('alex-snd/TRecover', model='trecover', device=device, version='latest')


def main() -> None:
    st.set_page_config(
        page_title='TRecover',
        page_icon='🩹',
        layout='wide',
        initial_sidebar_state='expanded')

    if 'history' not in st.session_state:
        st.session_state.history = list()

    if 'data' not in st.session_state:
        st.session_state.data = ''

    if 'regenerate' not in st.session_state:
        st.session_state.regenerate = False

    if 'columns' not in st.session_state:
        st.session_state.columns = None

    if 'is_unix' not in st.session_state:
        st.session_state.is_unix = platform.system() != 'Windows'

    sidebar()


def set_regenerate() -> None:
    st.session_state.regenerate = True


def unset_regenerate() -> None:
    st.session_state.regenerate = False


def sidebar() -> None:
    st.sidebar.header('Sections')

    option = st.sidebar.radio('', ('Inference', 'Inference history'))

    if option == 'Inference':
        is_plain, min_noise, max_noise, bw = inference_sidebar()
        inference_page(is_plain, min_noise, max_noise, bw)
    else:
        history_sidebar()
        history_page()


def inference_sidebar() -> Tuple[bool, int, int, int]:
    st.sidebar.text('\n')

    st.sidebar.header('Data parameters')

    data_type = st.sidebar.radio('Input type', ('Plain text', 'Noisy columns'), key='data_type',
                                 index=0 if 'Plain text' == st.session_state.get('data_type', 'Plain text') else 1)
    is_plain = data_type == 'Plain text'

    st.sidebar.text('\n')

    if is_plain:
        min_noise, max_noise = st.sidebar.slider('\nNoise range', 0, 25, key='noise_range',
                                                 value=st.session_state.get('noise_range', (0, 5)),
                                                 on_change=set_regenerate)
    else:
        min_noise, max_noise = 0, 0

    bw = st.sidebar.slider('Beam search width', 1, 26, key='beam_width',
                           value=st.session_state.get('beam_width', 5))

    if max_noise > var.MAX_NOISE:
        st.sidebar.warning('Max noise value is too large. This will entail poor performance')

    return is_plain, min_noise, max_noise + 1, bw


def history_sidebar() -> None:
    pass


def save_to_history(is_plain: bool,
                    min_noise: int,
                    max_noise: int,
                    bw: int,
                    columns: List[str],
                    chains: List[Tuple[str, float]]
                    ) -> None:
    text = st.session_state.data if is_plain else None

    st.session_state.history.append((is_plain, text, min_noise, max_noise, bw, columns, chains))


@st.cache(ttl=3600, show_spinner=False, suppress_st_warning=True)
def predict(columns: List[str], bw: int) -> List[Tuple[str, float]]:
    src = columns_to_tensor(columns, device)

    chains = beam_search(src, model, bw, device, beam_loop=dashboard_loop)
    chains = [(visualize_target(tensor_to_target(chain)), prob) for (chain, prob) in chains]

    return chains


def get_noisy_columns(data: str, min_noise: int, max_noise: int) -> List[str]:
    columns = create_noisy_columns(data, min_noise, max_noise)

    return [''.join(set(c)) for c in columns]  # kinda shuffle columns


def inference_page(is_plain: bool, min_noise: int, max_noise: int, bw: int) -> None:
    input_label = 'Insert plain text' if is_plain else 'Insert noisy columns separated by spaces'
    st.subheader(input_label)
    data = st.text_input('', value=st.session_state.data, max_chars=max_chars)

    if not data:
        st.stop()

    if is_plain:
        if st.session_state.regenerate or not st.session_state.columns or data != st.session_state.data:
            columns = get_noisy_columns(data, min_noise, max_noise)
            st.session_state.columns = columns
            unset_regenerate()
        else:
            columns = st.session_state.columns
    else:
        columns = data_to_columns(data, separator=' ')

    st.session_state.data = data

    st.subheader('\nColumns')
    st.text(visualize_columns(columns, delimiter=''))
    st.subheader('\n')

    placeholder = st.empty()
    recover_field, regen_filed = placeholder.columns([.07, 1])

    if is_plain:
        regen_filed.button('Regenerate', on_click=set_regenerate)

    if columns and recover_field.button('Recover'):
        if st.session_state.is_unix:
            with placeholder.container():
                progress_bar_placeholder = st.empty()
                st.button('Stop')

                with progress_bar_placeholder:
                    chains = predict(columns, bw)
        else:
            with placeholder:
                chains = predict(columns, bw)

        with placeholder.container():
            st.subheader('\nPrediction')
            st.text('\n\n'.join(chain for chain, _ in chains))

            if st.button('Clear'):
                st.session_state.task_id = None

        save_to_history(is_plain, min_noise, max_noise, bw, columns, chains)


def history_page() -> None:
    st.header('Inference History')

    if len(st.session_state.history) == 0:
        st.info('No records saved')
        return

    for record_id, (is_plain, text, min_noise, max_noise, bw, columns, chains) in enumerate(st.session_state.history,
                                                                                            start=1):
        st.info(f'Record {record_id}')

        if is_plain:
            st.text(f'Plain data: {text}')

        st.text(f'Noise range: [{min_noise}, {max_noise}]')
        st.text(f'Beam search width: {bw}')

        st.text('Columns:')
        st.text(visualize_columns(columns, delimiter=''))

        st.text('Prediction:')
        st.text('\n\n'.join(chain for chain, _ in chains))


if __name__ == '__main__':
    main()
