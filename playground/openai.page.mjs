import { client } from "./client.mjs";
import { html, create, styled } from "./misc.mjs";

import { OpenAI } from "openai";

import { useGlobalStore } from "./global.store.mjs";

const OpenaiPageContainer = styled.div`
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

const useStore = create((set, get) => ({
  params: {
    input: "你好，这是一个测试 [lbreak]",
    voice: "female2",
    style: "",
    speed: 1,
  },
  setParams: (params) =>
    set({
      params: {
        ...get().params,
        ...params,
      },
    }),
  result: null,
  setResult: (result) => set({ result }),
}));

const base_host =
  localStorage.getItem("__chattts_playground_api_base__") ||
  `${window.location.origin}`;
const openai = new OpenAI({
  apiKey: "sk-xxxx",
  baseURL: base_host + (base_host.endsWith("/") ? "" : "/") + "v1",
  dangerouslyAllowBrowser: true,
});

export const OpenaiPage = () => {
  // 可以传 input voice style 参数
  const { params, setParams, result, setResult } = useStore();
  const { input, voice, style, speed } = params;

  const { speakers, styles } = useGlobalStore();

  const handleInput = (e) => {
    setParams({ input: e.target.value });
  };

  const handleVoice = (e) => {
    if (e.target.value.startsWith("*")) {
      setParams({ voice: "" });
      return;
    }
    setParams({ voice: e.target.value });
  };

  const handleStyle = (e) => {
    if (e.target.value.startsWith("*")) {
      setParams({ style: "" });
      return;
    }
    setParams({ style: e.target.value });
  };

  const handleSpeed = (e) => {
    setParams({ speed: Number(e.target.value) });
  };

  const handleSubmit = async () => {
    const response = await openai.audio.speech.create({
      ...params,
      response_format: "mp3",
    });
    const blob = await response.blob();
    console.log(response);
    const url = URL.createObjectURL(blob);
    setResult(url);
  };

  return html`
    <${OpenaiPageContainer}>
      <fieldset>
        <legend>text</legend>
        <textarea value=${input} onInput=${handleInput}></textarea>
      </fieldset>
      <fieldset>
        <legend>Payload</legend>
        <label>
          Voice:
          <select value=${voice} onChange=${handleVoice}>
            <option value="-1">*random</option>
            ${speakers.map(
              (spk) => html`
                <option key=${spk.data.id} value=${spk.data.meta.data.name}>
                  ${spk.data.meta.data.name}
                </option>
              `
            )}
          </select>
        </label>
        <label>
          Style:
          <select value=${style} onChange=${handleStyle}>
            <option value="">*auto</option>
            ${styles.map(
              (style) => html`
                <option key=${style.id} value=${style.name}>
                  ${style.name}
                </option>
              `
            )}
          </select>
        </label>
        <label>
          速度
          <input
            type="range"
            min="0.5"
            max="1.5"
            step="0.1"
            value=${speed}
            onInput=${handleSpeed}
          />
          <output>${speed}</output>
        </label>
        <button onClick=${handleSubmit}>提交</button>
      </fieldset>
      <fieldset>
        <legend>Result</legend>
        ${result && html`<audio controls src=${result}></audio>`}
      </fieldset>
    <//>
  `;
};
