import { BaseClient } from "./client.mjs";

export class TtsClient extends BaseClient {
  /**
   *
   * @returns {Promise<{time: number, models: string[]}>}
   */
  async list_models() {
    const resp = await this._request("/v1/models/list");
    const { data } = await resp.json();
    return {
      models: data,
    };
  }

  /**
   *
   * @param {import("./types").ITTSParams} params
   */
  async tts(params) {
    const resp = await this._request("/v1/tts", params, "GET");
    const audio = await resp.arrayBuffer();
    return audio;
  }

  /**
   *
   * @param {import("./types").ITTSParamsV2} params
   */
  async tts_v2(params) {
    const resp = await this._request("/v2/tts", params, "POST");
    const audio = await resp.arrayBuffer();
    return audio;
  }
}
