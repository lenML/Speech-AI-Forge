import { client } from "./client.mjs";
import { html, create, styled } from "./misc.mjs";

import { OpenAI } from "openai";

import { useGlobalStore } from "./global.store.mjs";

const useStore = create((set, get) => ({
  payload: {
    input: {
      text: "你好，这是一个测试 [lbreak]",
      // ssml: "",
    },
    voice: {
      languageCode: "ZH-CN",
      name: "female2",
      style: "",
      temperature: 0.3,
      topP: 0.7,
      topK: 20,
      seed: 42,
    },
    audioConfig: {},
    enhancerConfig: {},
  },

  setPayload: (payload) => {
    set({
      input: {
        ...get().payload.input,
        ...payload.input,
      },
      voice: {
        ...get().payload.voice,
        ...payload.voice,
      },
      audioConfig: {
        ...get().payload.audioConfig,
        ...payload.audioConfig,
      },
      enhancerConfig: {},
    });
  },

  /**
   * audioContent: base64
   * @type {Array<{audioContent: string}>}
   */
  history: [],

  addToHistory: (item) => {
    set({ history: [...get().history, item] });
  },
}));

const Container = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;

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
    width: 100%;
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

  .payload-input {
    display: flex;
  }

  .history-items {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  }
`;

export const GooglePage = () => {
  const { payload, setPayload, history, addToHistory } = useStore();
  const { speakers, styles } = useGlobalStore();
  return html`
    <${Container}>
      <div className="payload-input">
        <fieldset
          style=${{
            flex: 2,
          }}
        >
          <legend>Input</legend>
          <textarea
            value=${payload.input.text}
            onChange=${(e) => setPayload({ input: { text: e.target.value } })}
          ></textarea>
        </fieldset>

        <fieldset
          style=${{
            flex: 1,
          }}
        >
          <legend>Voice</legend>
          <label>
            languageCode
            <select
              value=${payload.voice.languageCode}
              onChange=${(e) =>
                setPayload({ voice: { languageCode: e.target.value } })}
            >
              <option value="ZH-CN">ZH-CN</option>
              <option value="EN-US">EN-US</option>
            </select>
          </label>
          <label>
            name
            <select
              value=${payload.voice.name}
              onChange=${(e) => setPayload({ voice: { name: e.target.value } })}
            >
              ${speakers.map(
                (spk) =>
                  html` <option
                    key=${spk.data.id}
                    value=${spk.data.meta.data.name}
                  >
                    ${spk.data.meta.data.name}
                  </option>`
              )}
            </select>
          </label>

          <label>
            style
            <select
              value=${payload.voice.style}
              onChange=${(e) =>
                setPayload({ voice: { style: e.target.value } })}
            >
              <option value="">*auto</option>
              ${styles.map(
                (style) =>
                  html`
                    <option key=${style.id} value=${style.name}>
                      ${style.name}
                    </option>
                  `
              )}
            </select>
          </label>

          <label>
            temperature
            <input
              type="number"
              value=${payload.voice.temperature}
              onChange=${(e) =>
                setPayload({ voice: { temperature: e.target.value } })}
            />
          </label>
          <label>
            topP
            <input
              type="number"
              value=${payload.voice.topP}
              onChange=${(e) => setPayload({ voice: { topP: e.target.value } })}
            />
          </label>
          <label>
            topK
            <input
              type="number"
              value=${payload.voice.topK}
              onChange=${(e) => setPayload({ voice: { topK: e.target.value } })}
            />
          </label>
          <label>
            seed
            <input
              type="number"
              value=${payload.voice.seed}
              onChange=${(e) => setPayload({ voice: { seed: e.target.value } })}
            />
          </label>

          <button
            onClick=${async () => {
              const response = await client.googleTTTS(payload);
              addToHistory(response);
            }}
          >
            Generate
          </button>
        </fieldset>
      </div>

      <fieldset>
        <legend>History</legend>
        <div className="history-items">
          ${history
            .map((x, idx) => ({ idx, ...x }))
            .reverse()
            .map(
              (item, index) =>
                html` <div>
                  <span>${item.idx + 1}</span
                  ><audio controls src=${item.audioContent}></audio>
                </div>`
            )}
        </div>
      </fieldset>
    <//>
  `;
};
