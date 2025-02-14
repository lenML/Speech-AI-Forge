export class SAFClient {
  BASE_URL = "http://localhost:7870/";

  /**
   *
   * @param {string} base_url
   */
  constructor(base_url) {
    if (base_url) {
      this.BASE_URL = base_url;
    }
  }

  /**
   *
   * @param {string} path
   * @param {object} querys
   */
  #join(path, querys) {
    const url = new URL(path, this.BASE_URL);
    if (querys) {
      for (const [key, value] of Object.entries(querys)) {
        url.searchParams.append(key, value);
      }
    }
    return url.toString();
  }

  /**
   *
   * @param {string} path
   * @param {object | undefined} body
   * @param {"GET" | "POST" | undefined} method
   */
  async #request(path, body, method) {
    method = method ? method : body ? "POST" : "GET";
    const url = this.#join(path, method === "GET" ? body : undefined);
    const resp = await fetch(url, {
      method,
      ...(body && method === "POST"
        ? {
            headers: {
              "Content-Type": "application/json",
            },
            body: body ? JSON.stringify(body) : undefined,
          }
        : {}),
    });
    if (resp.status !== 200) {
      throw new Error(
        `Unexpected status code ${resp.status}.\n${await resp.text()}`
      );
    }
    return resp;
  }

  async ping() {
    const start = Date.now();
    await this.#request("/v1/ping");
    return {
      time: Date.now() - start,
    };
  }

  /**
   *
   * @returns {Promise<{time: number, models: string[]}>}
   */
  async list_models() {
    const resp = await this.#request("/v1/models/list");
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
    const resp = await this.#request("/v1/tts", params, "GET");
    const audio = await resp.arrayBuffer();
    return audio;
  }

  /**
   *
   * @param {import("./types").ITTSParamsV2} params
   */
  async tts_v2(params) {
    const resp = await this.#request("/v2/tts", params, "POST");
    const audio = await resp.arrayBuffer();
    return audio;
  }
}
