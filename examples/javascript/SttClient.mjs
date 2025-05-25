import fs from "fs";
import path from "path";
import { Blob } from "buffer";
import { BaseClient } from "./client.mjs";

/**
 *
 * @param {string} filepath
 * @returns
 */
function readFile(filepath) {
  const buffer = fs.readFileSync(filepath);
  return new Blob([buffer]);
}

/**
 * @typedef {Object} TranscriptionsForm
 * @property {string} model
 * @property {string} [prompt]
 * @property {string} [prefix]
 * @property {string} [language]
 * @property {number} [temperature]
 * @property {number} [sample_len]
 * @property {number} [best_of]
 * @property {number} [beam_size]
 * @property {number} [patience]
 * @property {number} [length_penalty]
 * @property {'txt'|'srt'|'vtt'|'json'} [format]
 * @property {boolean} [highlight_words]
 * @property {number} [max_line_count]
 * @property {number} [max_line_width]
 * @property {number} [max_words_per_line]
 */

/**
 * @typedef {Object} TranscriptionsResponseData
 * @property {string} text
 * @property {Array} segments
 * @property {Object} info
 */

/**
 * @typedef {Object} TranscriptionsResponse
 * @property {string} message
 * @property {TranscriptionsResponseData} data
 */

export class SttClient extends BaseClient {
  /**
   * 调用 STT 接口进行转录
   * @param {string} filePath 本地音频文件路径
   * @param {TranscriptionsForm} options 表单参数
   * @returns {Promise<TranscriptionsResponse>}
   */
  async transcribe(filePath, options) {
    const form = new FormData();
    form.append("file", readFile(filePath), path.basename(filePath));

    for (const [key, value] of Object.entries(options)) {
      if (value !== undefined && value !== null) {
        form.append(key, String(value));
      }
    }

    const resp = await fetch(this._join("/v1/stt/transcribe"), {
      method: "POST",
      body: form,
    });

    if (!resp.ok) {
      throw new Error(`STT error ${resp.status}: ${await resp.text()}`);
    }

    /** @type {TranscriptionsResponse} */
    const result = await resp.json();
    return result;
  }
}
