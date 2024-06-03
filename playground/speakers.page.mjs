import { client } from "./client.mjs";
import { html, create, styled } from "./misc.mjs";

/**
 * 管理 speaker
 *
 * 1. 显示 speaker 列表
 * 2. 创建 speaker
 * 3. 可以删除修改 speaker
 */
const useStore = create((set, get) => ({
  /**
   * @type {{ name: string, params: string }[]}
   */
  speakers: [],

  setSpeakers: (speakers) => set({ speakers }),

  formData: {
    seed: 42,
    name: "",
  },
  setFormData: (data) => set({ formData: data }),
}));

window.addEventListener("load", async () => {
  const speakers = await client.listSpeakers();
  useStore.get().setSpeakers(speakers.data);
});

const SpeakerFactory = () => {
  // 调用接口创建 speaker
  // 创建speaker需要设定seed和name
  const { setSpeakers, formData, setFormData } = useStore();
  return html`
    <feildset class="speaker-factory">
      <div>
        <label
          >seed

          <input
            type="number"
            value=${formData.seed}
            oninput=${(e) => setFormData({ ...formData, seed: e.target.value })}
          />
        </label>
        <label
          >name
          <input
            type="text"
            value=${formData.name}
            oninput=${(e) => setFormData({ ...formData, name: e.target.value })}
          />
        </label>

        <button
          onclick=${async () => {
            const speaker = await client.createSpeaker(formData);
            setSpeakers([...useStore.get().speakers, speaker]);
            setFormData({ seed: 0, name: "" });
          }}
        >
          创建
        </button>
      </div>
    </feildset>
  `;
};

const SpeakerList = () => {
  // 显示 speaker 列表
  // 只是可以列出来，没有其他操作
  const { speakers } = useStore();
  // 以table
  return html`
    <fieldset class="spekaer-list">
      <legend>Speakers</legend>

      <table class="speaker-list">
        <thead>
          <tr>
            <th>id</th>
            <th>name</th>
          </tr>
        </thead>
        <tbody>
          ${speakers.map(
            (speaker) => html`
              <tr>
                <td>${speaker.id}</td>
                <td>${speaker.name}</td>
              </tr>
            `
          )}
        </tbody>
      </table>
    </fieldset>
  `;
};

const SpeakerPageContainer = styled.div`
  display: flex;
  flex-direction: row;

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

  th:nth-child(2),
  td:nth-child(2) {
    width: 60%;
  }

  .speaker-factory {
    flex: 1;
  }

  .spekaer-list {
    width: 256px;
  }
`;

export const SpeakerPage = () => {
  return html`
    <${SpeakerPageContainer}>
      <${SpeakerList} />
      <${SpeakerFactory} />
    <//>
  `;
};
