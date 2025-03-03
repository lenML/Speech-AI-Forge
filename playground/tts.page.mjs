import { client } from "./client.mjs";
import { html, create, styled } from "./misc.mjs";

import { useGlobalStore } from "./global.store.mjs";

const sample_texts = [
  {
    text: "大🍌，一条大🍌，嘿，你的感觉真的很奇妙  [lbreak]",
  },
  {
    text: "天气预报显示，今天会有小雨，请大家出门时记得带伞。降温的天气也提醒我们要适时添衣保暖。 [lbreak]",
  },
  {
    text: "公司的年度总结会议将在下周三举行，请各部门提前准备好相关材料，确保会议顺利进行。 [lbreak]",
  },
  {
    text: "今天的午餐菜单包括烤鸡、沙拉和蔬菜汤，大家可以根据自己的口味选择适合的菜品。 [lbreak]",
  },
  {
    text: "请注意，电梯将在下午两点进行例行维护，预计需要一个小时的时间，请大家在此期间使用楼梯。 [lbreak]",
  },
  {
    text: "图书馆新到了一批书籍，涵盖了文学、科学和历史等多个领域，欢迎大家前来借阅。 [lbreak]",
  },
  {
    text: "电影中梁朝伟扮演的陈永仁的编号27149 [lbreak]",
  },
  {
    text: "这块黄金重达324.75克 [lbreak]",
  },
  {
    text: "我们班的最高总分为583分 [lbreak]",
  },
  {
    text: "12~23 [lbreak]",
  },
  {
    text: "-1.5~2 [lbreak]",
  },
  {
    text: "她出生于86年8月18日，她弟弟出生于1995年3月1日 [lbreak]",
  },
  {
    text: "等会请在12:05请通知我 [lbreak]",
  },
  {
    text: "今天的最低气温达到-10°C [lbreak]",
  },
  {
    text: "现场有7/12的观众投出了赞成票 [lbreak]",
  },
  {
    text: "明天有62％的概率降雨 [lbreak]",
  },
  {
    text: "随便来几个价格12块5，34.5元，20.1万 [lbreak]",
  },
  {
    text: "这是固话0421-33441122 [lbreak]",
  },
  {
    text: "这是手机+86 18544139121 [lbreak]",
  },
];

let history_index = 0;

const useStore = create((set, get) => ({
  tts: {
    text: "你好，这里是一段ChatTTS Forge项目的示例文本。",
    spk: "female2",
    style: "chat",
    temperature: 0.3,
    top_p: 0.5,
    top_k: 20,
    seed: 42,
    format: "mp3",
    prompt1: "",
    prompt2: "",
    prefix: "",
  },

  ui: {
    loading: false,
    // 历史生成结果 { audio: Blob, url: string, params: object }
    history: [],
  },

  async synthesizeTTS() {
    const params = structuredClone(get().tts);
    const blob = await client.synthesizeTTS({
      ...params,
    });
    const blob_url = URL.createObjectURL(blob);
    set({
      ui: {
        ...get().ui,
        history: [
          ...get().ui.history,
          {
            id: history_index++,
            audio: blob,
            url: blob_url,
            params: params,
          },
        ],
      },
    });
  },
  setTTS(tts) {
    set({
      tts: {
        ...get().tts,
        ...tts,
      },
    });
  },
  setUI(ui) {
    set({
      ui: {
        ...get().ui,
        ...ui,
      },
    });
  },
}));

const TTSPageContainer = styled.div`
  h1 {
    margin-bottom: 1rem;
  }

  p {
    margin-bottom: 1rem;
  }

  #app {
    margin-top: 1rem;
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

  .content-body {
    display: flex;
    gap: 1rem;
  }

  .content-left {
    flex: 1;
  }

  .content-right {
    flex: 4;
  }

  h1 small {
    font-weight: 100;
    font-size: 0.5em;
    font-weight: normal;
  }

  .btn-synthesize {
    background-color: #007bff;
    color: white;
    border: none;
    cursor: pointer;
    padding: 0.5rem 1rem;
  }

  .btn-synthesize:hover {
    background-color: #0056b3;
  }

  .btn-synthesize:disabled {
    background-color: #6c757d;
    cursor: not-allowed;
  }

  .btn-clear {
    background-color: #dc3545;
    color: white;
    border: none;
    cursor: pointer;
    padding: 0.5rem 1rem;
  }

  .btn-clear:hover {
    background-color: #bd2130;
  }

  .btn-clear:disabled {
    background-color: #6c757d;
    cursor: not-allowed;
  }

  .btn-random {
    background-color: #28a745;
    color: white;
    border: none;
    cursor: pointer;
    padding: 0.5rem 1rem;
  }

  .btn-random:hover {
    background-color: #218838;
  }

  pre {
    white-space: pre-wrap;
  }

  .sample-texts {
    width: unset;
    display: inline-block;
    padding: 0.5rem;
    margin-bottom: 1rem;
  }
`;

export const TTSPage = () => {
  const { tts, setTTS, synthesizeTTS, ui, setUI } = useStore();
  const { speakers, styles, formats } = useGlobalStore();

  const request = async () => {
    if (ui.loading) {
      return;
    }
    setUI({ loading: true });
    try {
      await synthesizeTTS();
    } catch (error) {
      alert(error);
      console.error("Error synthesizing TTS:", error);
    } finally {
      setUI({ loading: false });
    }
  };

  return html`
    <${TTSPageContainer}>
      <textarea
        value=${tts.text}
        onInput=${(e) => setTTS({ text: e.target.value })}
      ></textarea>
      <button class="btn-synthesize" disabled=${ui.loading} onClick=${request}>
        ${ui.loading ? "Synthesizing..." : "Synthesize"}
      </button>
      <button
        class="btn-clear"
        disabled=${ui.loading}
        onClick=${() => setUI({ history: [] })}
      >
        Clear History
      </button>

      <select
        placeholder="Sample Text"
        class="sample-texts"
        value=${tts.text}
        onChange=${(e) => setTTS({ text: e.target.value })}
      >
        ${sample_texts.map(
          (item, index) => html`
            <option key=${index} value=${item.text}>
              Sample ${index + 1}: ${item.text.slice(0, 10) + "..."}
            </option>
          `
        )}
      </select>

      <div class="content-body">
        <fieldset class="content-left">
          <legend>Options</legend>
          <label>
            Speaker:
            <select
              value=${tts.spk}
              onChange=${(e) => setTTS({ spk: e.target.value })}
            >
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
            <select
              value=${tts.style}
              onChange=${(e) => setTTS({ style: e.target.value })}
            >
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
            Temperature:
            <input
              type="range"
              min="0.01"
              max="2"
              step="0.01"
              value=${tts.temperature}
              onInput=${(e) => setTTS({ temperature: e.target.value })}
            />
            ${tts.temperature}
          </label>
          <label>
            Top P:
            <input
              type="range"
              min="0.01"
              max="1"
              step="0.01"
              value=${tts.top_p}
              onInput=${(e) => setTTS({ top_p: e.target.value })}
            />
            ${tts.top_p}
          </label>
          <label>
            Top K:
            <input
              type="range"
              min="1"
              max="50"
              step="1"
              value=${tts.top_k}
              onInput=${(e) => setTTS({ top_k: e.target.value })}
            />
            ${tts.top_k}
          </label>
          <label>
            Seed:
            <input
              type="number"
              value=${tts.seed}
              onInput=${(e) => setTTS({ seed: e.target.value })}
            />
            <button
              class="btn-random"
              onClick=${() =>
                setTTS({ seed: Math.floor(Math.random() * 2 ** 32 - 1) })}
            >
              Random
            </button>
          </label>
          <label>
            Format
            <select
              value=${tts.format}
              onChange=${(e) => setTTS({ format: e.target.value })}
            >
              ${formats.map(
                (format) =>
                  html`<option key=${format} value=${format}>${format}</option>`
              )}
            </select>
          </label>
          <label>
            Prompt1:
            <input
              type="text"
              value=${tts.prompt1}
              onInput=${(e) => setTTS({ prompt1: e.target.value })}
            />
          </label>
          <label>
            Prompt2:
            <input
              type="text"
              value=${tts.prompt2}
              onInput=${(e) => setTTS({ prompt2: e.target.value })}
            />
          </label>
          <label>
            Prefix:
            <input
              type="text"
              value=${tts.prefix}
              onInput=${(e) => setTTS({ prefix: e.target.value })}
            />
          </label>
        </fieldset>

        <fieldset class="content-right">
          <legend>History</legend>
          <table>
            <thead>
              <tr>
                <th>id</th>
                <th>Params</th>
                <th>Audio</th>
              </tr>
            </thead>
            <tbody>
              ${[...ui.history].reverse().map(
                (item, index) => html`
                  <tr key=${item.id}>
                    <td>${item.id}</td>
                    <td>
                      <pre>${JSON.stringify(item.params, null, 2)}</pre>
                    </td>
                    <td>
                      <audio controls>
                        <source
                          src=${item.url}
                          type="audio/${{
                            raw: "wav",
                          }[item.params.format] || item.params.format}"
                        />
                      </audio>
                    </td>
                  </tr>
                `
              )}
            </tbody>
          </table>
        </fieldset>
      </div>
    <//>
  `;
};
