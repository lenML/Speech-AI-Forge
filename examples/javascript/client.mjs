export class BaseClient {
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
  _join(path, querys) {
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
  async _request(path, body, method) {
    method = method ? method : body ? "POST" : "GET";
    const url = this._join(path, method === "GET" ? body : undefined);
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
    await this._request("/v1/ping");
    return {
      time: Date.now() - start,
    };
  }
}
