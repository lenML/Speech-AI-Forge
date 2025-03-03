import { client } from "./client.mjs";
import { html, create, styled } from "./misc.mjs";

import { useGlobalStore } from "./global.store.mjs";

const default_ssml = `
<speak version="0.1">
  <voice spk="Bob" seed="-1" style="narration-relaxed">
    这里是一个简单的 SSML 示例。 [lbreak]
  </voice>
</speak>
`.trim();

const useStore = create((set, get) => ({
  params: {
    ssml: default_ssml,
    format: "mp3",
  },
  setParams: (params) => set({ params }),

  loading: false,

  /**
   * @type {Array<{ id: number, params: { ssml: string; format: string }, url: string }>}
   */
  history: [],
  setHistory: (history) => set({ history }),
}));

const SSMLFormContainer = styled.div`
  display: flex;
  flex-direction: column;

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

  .ssml-body {
    display: flex;
    gap: 1rem;
  }

  table {
    width: 100%;
    border-collapse: collapse;
  }

  th,
  td {
    padding: 0.5rem;
    border: 1px solid #333;
  }

  th {
    background-color: #333;
    color: white;
  }

  .btn-danger {
    background-color: #dc3545;
    color: white;
    border: none;
    cursor: pointer;
  }

  .btn-danger:hover {
    background-color: #bd2130;
  }
`;

const SSMLOptions = () => {
  const { params, setParams } = useStore();
  const { formats } = useGlobalStore();
  return html`
    <fieldset style=${{ flex: 5 }}>
      <legend>Options</legend>
      <label>
        Format
        <select
          value=${params.format}
          onChange=${(e) => setParams({ ...params, format: e.target.value })}
        >
          ${formats.map(
            (format) =>
              html` <option key=${format} value=${format}>${format}</option> `
          )}
        </select>
      </label>
    </fieldset>
  `;
};

const SSMLHistory = () => {
  const { history } = useStore();
  return html`
    <fieldset style=${{ flex: 5 }}>
      <legend>History</legend>

      <table>
        <thead>
          <tr>
            <th>index</th>
            <th>SSML</th>
            <th>Audio</th>
          </tr>
        </thead>
        <tbody>
          ${[...history].reverse().map(
            (item) => html`
              <tr key=${item.id}>
                <td>${item.id}</td>
                <td>
                  <textarea
                    readonly
                    style=${{
                      with: "100%",
                      height: "5rem",
                      resize: "none",
                    }}
                  >
                    ${item.params.ssml}
                  </textarea
                  >
                </td>
                <td>
                  <audio controls>
                    <source
                      src=${item.url}
                      type="audio/${{ raw: "wav" }[item.params.format] ||
                      item.params.format}"
                    />
                  </audio>
                </td>
              </tr>
            `
          )}
        </tbody>
      </table>
    </fieldset>
  `;
};

let generate_index = 0;

const SSMLForm = () => {
  const { params, setParams, loading } = useStore();
  const request = async () => {
    useStore.set({ loading: true });
    try {
      const blob = await client.synthesizeSSML(params);
      const blob_url = URL.createObjectURL(blob);
      useStore.set({
        history: [
          ...useStore.get().history,
          {
            id: generate_index++,
            params,
            url: blob_url,
          },
        ],
      });
    } catch(err) {
      alert(err);
      console.error(err);
    } finally {
      useStore.set({ loading: false });
    }
  };
  return html`
    <${SSMLFormContainer}>
      <textarea
        placeholder="Enter SSML here..."
        value=${params.ssml}
        onInput=${(e) => setParams({ ...params, ssml: e.target.value })}
      />
      <div>
        <button onClick=${request} disabled=${!params.ssml || loading}>
          Submit
        </button>
        <button
          class="btn btn-danger"
          onClick=${() => {
            useStore.set({ history: [] });
          }}
          disabled=${loading}
        >
          Clear History
        </button>
      </div>

      <div class="ssml-body">
        <${SSMLOptions} />
        <${SSMLHistory} />
      </div>
    <//>
  `;
};

const SSMLPageContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
`;

export const SSMLPage = () => {
  return html` <${SSMLPageContainer}>
    <${SSMLForm} />
  <//>`;
};
