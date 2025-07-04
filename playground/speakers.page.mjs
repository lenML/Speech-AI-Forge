import { client } from "./client.mjs";
import { html, create, styled } from "./misc.mjs";

import {
  useReactTable,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  flexRender,
  createColumnHelper,
} from "@tanstack/react-table";

import * as React from "react";
import { deepEqual } from "fast-equals";

import { default as _ } from "lodash";

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

  setSpeakers: (speakers) => set({ speakers, temp_speakers: [...speakers] }),

  // 编辑时的 speaker
  temp_speakers: [],
  setTempSpeakers: (speakers) => set({ temp_speakers: speakers }),

  formData: {
    seed: 42,
    name: "",
  },
  setFormData: (data) => set({ formData: data }),
}));

window.addEventListener("load", async () => {
  const speakers = await client.listSpeakers();
  useStore.get().setSpeakers(speakers.data.items);
});

const CreateForm = () => {
  const { setSpeakers, formData, setFormData } = useStore();
  // 还可以设置 gender describe
  return html`
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
      <label>
        gender
        <select onChnage=${(e) => {
          setFormData({ ...formData, gender: e.target.value });
        }}>
        <option value="*">*</opetion>
        <option value="female">female</opetion>
          <option value="male">male</opetion>
        </select>
      </label>
      <label>
        describe
        <textarea></textarea>
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
  `;
};

const SpeakerFactory = () => {
  return html`
    <fieldset class="speaker-factory">
      <legend>Speaker Factory</legend>
      <${CreateForm} />
    </fieldset>
  `;
};

// Give our default column cell renderer editing superpowers!
const defaultColumn = {
  cell: ({ getValue, row: { index }, column: { id }, table }) => {
    const initialValue = getValue();
    // We need to keep and update the state of the cell normally
    const [value, setValue] = React.useState(initialValue);

    // When the input is blurred, we'll call our table meta's updateData function
    const onBlur = () => {
      table.options.meta?.updateData(index, id, value);
    };

    // If the initialValue is changed external, sync it up with our state
    React.useEffect(() => {
      setValue(initialValue);
    }, [initialValue]);

    return html`<input
      value=${value}
      onChange=${(e) => setValue(e.target.value)}
      onBlur=${onBlur}
    />`;
  },
};

const columnHelper = createColumnHelper();
const columns = [
  columnHelper.accessor("data.id", {
    header: "ID",
    cell: (info) => html`<div className="td-id">${info.getValue()}</div>`,
  }),
  columnHelper.accessor("data.meta.data.name", {
    header: "Name",
  }),
  columnHelper.accessor("data.meta.data.gender", {
    header: "Gender",
  }),
  columnHelper.accessor("data.meta.data.desc", {
    header: "Describe",
  }),
  columnHelper.accessor("data.meta.data.author", {
    header: "Author",
  }),
  columnHelper.accessor("data.meta.data.version", {
    header: "Version",
  }),
];

function Filter({ column, table }) {
  const firstValue = table
    .getPreFilteredRowModel()
    .flatRows[0]?.getValue(column.id);

  const columnFilterValue = column.getFilterValue();

  return typeof firstValue === "number"
    ? html`<div className="flex space-x-2">
        <input
          type="number"
          value=${columnFilterValue?.[0] ?? ""}
          onChange=${(e) =>
            column.setFilterValue((old) => [e.target.value, old?.[1]])}
          placeholder=${`Min`}
        />
        <input
          type="number"
          value=${columnFilterValue?.[1] ?? ""}
          onChange=${(e) =>
            column.setFilterValue((old) => [old?.[0], e.target.value])}
          placeholder=${`Max`}
        />
      </div>`
    : html`<input
        type="text"
        value=${columnFilterValue ?? ""}
        onChange=${(e) => column.setFilterValue(e.target.value)}
        placeholder=${`Search...`}
      />`;
}
function useSkipper() {
  const shouldSkipRef = React.useRef(true);
  const shouldSkip = shouldSkipRef.current;

  // Wrap a function with this to skip a pagination reset temporarily
  const skip = React.useCallback(() => {
    shouldSkipRef.current = false;
  }, []);

  React.useEffect(() => {
    shouldSkipRef.current = true;
  });

  return [shouldSkip, skip];
}

const PaginationContainer = styled.div`
  display: flex;
  padding: 8px;
  gap: 8px;
  height: 56px;

  .divider {
    flex: 1;
  }

  input {
    width: auto;
    display: inline-block;
  }
  select {
    width: auto;
    display: inline-block;
  }

  & > div {
    display: inline-flex;
    white-space: nowrap;
    justify-content: center;
    align-items: center;
  }
`;

const SpeakerTable = () => {
  // 显示 speaker 列表
  // 只是可以列出来，没有其他操作
  const { speakers, temp_speakers, setSpeakers, setTempSpeakers } = useStore();

  const dataChanged = React.useMemo(() => {
    return !deepEqual(speakers, temp_speakers);
  });

  const [autoResetPageIndex, skipAutoResetPageIndex] = useSkipper();

  const table = useReactTable({
    data: temp_speakers,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    autoResetPageIndex,
    defaultColumn: defaultColumn,
    meta: {
      updateData: (rowIndex, columnId, value) => {
        // Skip page index reset until after next rerender
        skipAutoResetPageIndex();
        const old = useStore.get().temp_speakers;
        setTempSpeakers(
          old.map((row, index) => {
            if (index === rowIndex) {
              row = _.cloneDeep(row);
              _.set(row, columnId.replace("data_", "data."), value);
              return row;
            }
            return row;
          })
        );
      },
    },
  });

  const handleReset = React.useCallback(() => {
    setTempSpeakers(speakers);
  }, [speakers]);
  const handleSave = React.useCallback(() => {
    setSpeakers(temp_speakers);

    client.updateSpeakers({
      speakers: temp_speakers,
    });
  }, [temp_speakers]);

  return html`
    <fieldset class="spekaer-list">
      <legend>Speakers</legend>
      <table class="speaker-table">
        <thead>
          ${table.getHeaderGroups().map(
            (headerGroup) =>
              html`<tr key="${headerGroup.id}">
                ${headerGroup.headers.map(
                  (header) =>
                    html`<th key=${header.id} colspan=${header.colSpan}>
                      ${header.isPlaceholder
                        ? null
                        : html`<div>
                            ${flexRender(
                              header.column.columnDef.header,
                              header.getContext()
                            )}
                            ${header.column.getCanFilter()
                              ? html`<div>
                                  <${Filter}
                                    column=${header.column}
                                    table=${table}
                                  />
                                </div>`
                              : null}
                          </div>`}
                    </th>`
                )}
              </tr>`
          )}
        </thead>
        <tbody>
          ${table.getRowModel().rows.map(
            (row) =>
              html`<tr key="${row.id}">
                ${row
                  .getVisibleCells()
                  .map(
                    (cell) =>
                      html`<td key="${cell.id}">
                        ${flexRender(
                          cell.column.columnDef.cell,
                          cell.getContext()
                        )}
                      </td>`
                  )}
              </tr>`
          )}
        </tbody>
      </table>
      <${PaginationContainer}>
        <button
          onClick=${() => table.setPageIndex(0)}
          disabled=${!table.getCanPreviousPage()}
        >
          ${"<<"}
        </button>
        <button
          onClick=${() => table.previousPage()}
          disabled=${!table.getCanPreviousPage()}
        >
          ${"<"}
        </button>
        <button
          onClick=${() => table.nextPage()}
          disabled=${!table.getCanNextPage()}
        >
          ${">"}
        </button>
        <button
          onClick=${() => table.setPageIndex(table.getPageCount() - 1)}
          disabled=${!table.getCanNextPage()}
        >
          ${">>"}
        </button>
        <div className="divider"></div>
        <div>
          <span>Page</span>
          <strong>
            ${table.getState().pagination.pageIndex + 1} of ${" "}
            ${table.getPageCount()}
          </strong>
        </div>
        <select
          value=${table.getState().pagination.pageSize}
          onChange=${(e) => {
            table.setPageSize(Number(e.target.value));
          }}
        >
          ${[10, 20, 30, 40, 50].map(
            (pageSize) =>
              html`<option key=${pageSize} value=${pageSize}>
                Show ${pageSize}
              </option>`
          )}
        </select>
      <//>
      <button
        className="btn-save"
        disabled=${!dataChanged}
        onClick=${handleSave}
      >
        save
      </button>
      <button
        className="btn-reset"
        disabled=${!dataChanged}
        onClick=${handleReset}
      >
        reset
      </button>
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

  tbody tr:nth-child(odd) {
    background-color: #111;
  }

  .speaker-factory {
    flex: 1;
  }

  .spekaer-list {
    flex: 3;
  }

  .speaker-table {
    width: 100%;
    border-collapse: collapse;

    td,
    th {
      padding: 8px;
      text-align: left;
    }

    th:first-child,
    td:first-child {
      & > * {
        width: 5rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
    }

    th:not(:first-child),
    td:not(:first-child) {
      width: 30%;
    }

    input {
      width: 100%;
      background-color: transparent;
      border-color: transparent;

      &:focus {
        background-color: #333;
        border-color: #333;
      }
    }
  }

  .btn-save:not(:disabled) {
    background-color: #28a745;
  }
  .btn-save:not(:disabled):hover {
    background-color: #218838;
  }
`;

export const SpeakerPage = () => {
  return html`
    <${SpeakerPageContainer}>
      <${SpeakerFactory} />
      <${SpeakerTable} />
    <//>
  `;
};
