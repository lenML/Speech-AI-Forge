import { BaseClient } from "./client.mjs";

/**
 * @template T
 * @typedef {import('./types').Speaker.BaseResponse<T>} BaseResponse
 */

export class SpkClient extends BaseClient {
  /**
   * @param {import('./types').Speaker.SpkListParams} params
   *  @returns {Promise<BaseResponse<import('./types').Speaker.PaginatedResponse<import('./types').Speaker.Speaker>>>}
   */
  async list(params) {
    const res = await this._request("/v1/speakers/list", params, "GET");
    return res.json();
  }

  /** @returns {Promise<BaseResponse<null>>} */
  async refresh() {
    const res = await this._request("/v1/speakers/refresh");
    return res.json();
  }

  /**
   * @param {import('./types').Speaker.SpeakersUpdate} data
   *  @returns {Promise<BaseResponse<null>>}
   */
  async batchUpdate(data) {
    const res = await this._request("/v1/speakers/update", data);
    return res.json();
  }

  /**
   * @param {import('./types').Speaker.CreateSpeaker} data
   *  @returns {Promise<BaseResponse<import('./types').Speaker.Speaker>>}
   */
  async create(data) {
    const res = await this._request("/v1/speaker/create", data);
    return res.json();
  }

  /**
   * @param {import('./types').Speaker.UpdateSpeaker} data
   *  @returns {Promise<BaseResponse<null>>}
   */
  async update(data) {
    const res = await this._request("/v1/speaker/update", data);
    return res.json();
  }

  /**
   * @param {import('./types').Speaker.SpeakerDetail} data
   *  @returns {Promise<BaseResponse<import('./types').Speaker.Speaker>>}
   */
  async detail(data) {
    const res = await this._request("/v1/speaker/detail", data);
    return res.json();
  }
}

// (async () => {
//   const client = new SpkClient();
//   const res = await client.list({ limit: 5, offset: 0 });
//   console.log(res);
// })();
