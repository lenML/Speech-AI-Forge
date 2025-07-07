import { client } from "./client.mjs";
import { html, create, styled } from "./misc.mjs";

import { useGlobalStore } from "./global.store.mjs";

const StreamPageContainer = styled.div`
  display: flex;
  width: 100%;

  & > * {
    flex: 1;
  }

  textarea {
    width: 100%;
    height: 10rem;
    margin-bottom: 1rem;

    min-height: 10rem;

    resize: vertical;
  }

  button {
    padding: 0.5rem 1rem;
    background-color: #007bff;
    color: white;
    border: none;
    cursor: pointer;
  }

  button:hover {
    background-color: #0056b3;
  }

  button:disabled {
    background-color: #6c757d;
    cursor: not-allowed;
  }

  fieldset {
    margin-top: 1rem;
    padding: 1rem;
    border: 1px solid #333;
  }

  legend {
    font-weight: bold;
  }

  label {
    display: block;
    margin-bottom: 0.5rem;
  }

  select,
  input[type="range"],
  input[type="number"] {
    width: 100%;
    margin-top: 0.25rem;
  }

  input[type="range"] {
    width: calc(100% - 2rem);
  }

  input[type="number"] {
    width: calc(100% - 2rem);
    padding: 0.5rem;
  }

  input[type="text"] {
    width: 100%;
    padding: 0.5rem;
  }

  audio {
    margin-top: 1rem;
  }

  textarea,
  input,
  select {
    background-color: #333;
    color: white;
    border: 1px solid #333;
    border-radius: 0.25rem;
    padding: 0.5rem;
  }
`;

const support_models = [
  "chat-tts",
  "fish-speech",
  "cosy-voice",
  "fire-red-tts",
  "f5-tts",
  "index-tts",
  "spark-tts",
  "gpt-sovits-v1",
  "gpt-sovits-v2",
  "gpt-sovits-v3",
  "gpt-sovits-v4",
];
const useStore = create((set, get) => ({
  params: {
    text: "你好，这是一个测试。你好，这是一个测试。你好，这是一个测试。你好，这是一个测试。你好，这是一个测试。你好，这是一个测试。",
    spk: "female2",
    style: "",
    temperature: 0.3,
    top_p: 0.5,
    top_k: 20,
    seed: 42,
    format: "mp3",
    no_cache: false,
    model: "chat-tts",
  },
  setParams: (params) =>
    set({
      params: {
        ...get().params,
        ...params,
      },
    }),

  audio_url: "",

  setAudioUrl: (audio_url) => set({ audio_url }),
}));

const StreamForm = () => {
  const { speakers, styles, formats } = useGlobalStore();

  const { params, setParams, setAudioUrl } = useStore();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setParams({ [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const {
      text,
      spk,
      style,
      temperature,
      top_p,
      top_k,
      seed,
      format,
      no_cache,
      model,
    } = params;
    const audio_url = client.synthesizeTTSUrl({
      text,
      spk,
      style,
      temperature,
      top_p,
      top_k,
      seed,
      format,
      no_cache,
      stream: true,
      model,
    });
    setAudioUrl(audio_url);
  };

  return html`
    <fieldset>
      <legend>Stream Form</legend>
      <form onSubmit=${handleSubmit}>
        <label>
          Text
          <textarea name="text" value=${params.text} onChange=${handleChange} />
        </label>
        <label>
          Model
          <select name="model" value=${params.model} onChange=${handleChange}>
            ${support_models.map(
              (name) => html`<option key=${name} value=${name}>${name}</option>`
            )}
          </select>
        </label>
        <label>
          Speaker
          <select name="spk" value=${params.spk} onChange=${handleChange}>
            ${speakers.map(
              (spk) =>
                html`<option
                  key=${spk.data.id}
                  value=${spk.data.meta.data.name}
                >
                  ${spk.data.meta.data.name}
                </option>`
            )}
          </select>
        </label>
        <label>
          Style
          <select name="style" value=${params.style} onChange=${handleChange}>
            <option value=${""}>*auto</option>
            ${styles.map(
              (style) =>
                html`<option key=${style.id} value=${style.name}>
                  ${style.name}
                </option>`
            )}
          </select>
        </label>
        <label>
          Temperature
          <input
            type="number"
            name="temperature"
            value=${params.temperature}
            onChange=${handleChange}
          />
        </label>
        <label>
          Top P
          <input
            type="number"
            name="top_p"
            value=${params.top_p}
            onChange=${handleChange}
          />
        </label>
        <label>
          Top K
          <input
            type="number"
            name="top_k"
            value=${params.top_k}
            onChange=${handleChange}
          />
        </label>
        <label>
          Seed
          <input
            type="number"
            name="seed"
            value=${params.seed}
            onChange=${handleChange}
          />
        </label>
        <label>
          Format
          <select name="format" value=${params.format} onChange=${handleChange}>
            ${formats.map(
              (format) =>
                html`<option key=${format} value=${format}>${format}</option>`
            )}
          </select>
        </label>
        <label>
          No cache
          <input
            type="checkbox"
            name="no_cache"
            checked=${params.no_cache}
            onChange=${handleChange}
          />
        </label>
        <button type="submit">Synthesize</button>
      </form>
    </fieldset>
  `;
};

const StreamAudio = () => {
  const { audio_url } = useStore();
  return html`
    <fieldset>
      <legend>Stream Audio</legend>
      <audio controls src=${audio_url} />
    </fieldset>
  `;
};

export const StreamPage = () => {
  return html` <${StreamPageContainer}>
    <${StreamForm} />
    <${StreamAudio} />
  <//>`;
};
