import platform
from typing import Tuple, List, Dict, Optional

import streamlit as st
import torch
from trecover.config import var
from trecover.utils.beam_search import beam_search, dashboard_loop
from trecover.utils.inference import data_to_columns, create_noisy_columns
from trecover.utils.transform import columns_to_tensor, tensor_to_target
from trecover.utils.visualization import visualize_columns, visualize_target

MAX_CHARS = 256

PLAIN_EXAMPLES = {
    'Select example': None,
    'Example 1': 'As people around the country went into the streets to cheer the conviction, some businesses in '
                 'Portland boarded up their windows once again.',
    'Example 2': 'That night, a small group of activists wearing black approached a group of journalists, threatening'
                 ' to smash the cameras of those who remained on scene.',
    'Example 3': 'English as we know it today came to be exported to other parts of the world through British '
                 'colonisation, and is now the dominant language in Britain'
}

NOISED_EXAMPLES = {
    'Select example': None,
    'Example 1': 'a ds fpziq ofe ngkhbo p pghl ue waq frlqjo o u dnxrm dgr yrtsco kho deuasm dhysc ao u nwzhy tle r '
                 'yzpe xwabc gce nger klqto wiq nfprso t no tpgq tcfh ae twas tw ur re e t gyutsm t xgo rc ubhq e wle '
                 'r ty h nwpeaq xdsc o dnhelm v thir ikcq tkuo i o twn ps frio mo oe b kuiqtb jsq zi tnye ge dgrqs s '
                 'cioe ys whic wne wp thlo dnprsc xvpyrt hurlm kveaj nbfp dome pbeaj dusmo a r dzrqsm xace du nxkuai '
                 'gpulcm tpi h pie uim r wbhrj ui n dwgp dkeio nkwhqs zs'
}

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
    st.sidebar.markdown(body=
                        """
                        <h1 align="center"> 
                            <font size="20">🤷</font>
                            <a href="https://alex-snd.github.io/TRecover">About the Project</a>
                        </h1>
                        <br><br>
                        """,
                        unsafe_allow_html=True)

    option = st.sidebar.radio('Sections', ('Inference', 'Inference history'))

    if option == 'Inference':
        is_plain, min_noise, max_noise, bw = inference_sidebar()
        inference_page(is_plain, min_noise, max_noise, bw)
    else:
        history_sidebar()
        history_page()


def inference_sidebar() -> Tuple[bool, int, int, int]:
    st.sidebar.text('\n')

    data_type = st.sidebar.radio('Input type', ('Plain text', 'Noisy columns'), key='data_type',
                                 index=0 if 'Plain text' == st.session_state.get('data_type', 'Plain text') else 1)
    is_plain = data_type == 'Plain text'

    st.sidebar.text('\n')

    if is_plain:
        min_noise, max_noise = st.sidebar.slider('\nNoise range', 0, 5, key='noise_range',
                                                 value=st.session_state.get('noise_range', (0, 5)),
                                                 on_change=set_regenerate)
    else:
        min_noise, max_noise = 0, 0

    bw = st.sidebar.slider('Beam search width', 1, 6, key='beam_width',
                           value=st.session_state.get('beam_width', 25))

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


def get_input_data(examples: Dict[str, Optional[str]], max_chars: int) -> str:
    input_field, examples_filed = st.columns([1, 0.27])

    option = examples_filed.selectbox(label='', options=examples.keys())

    return input_field.text_input(label='', value=examples[option] or st.session_state.data, max_chars=max_chars)


def inference_page(is_plain: bool, min_noise: int, max_noise: int, bw: int) -> None:
    st.subheader('Insert plain text' if is_plain else 'Insert noisy columns separated by spaces')

    if is_plain:
        data = get_input_data(PLAIN_EXAMPLES, max_chars=MAX_CHARS)
    else:
        data = get_input_data(NOISED_EXAMPLES, max_chars=MAX_CHARS * 4)

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
    recover_field, regen_filed = placeholder.columns([.11, 1])

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