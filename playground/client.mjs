import axios from "axios";

/**
 *
 * @param {string} baseURL - The base URL to join with the paths and search params
 * @param {string[]} paths - The paths to join with the base URL
 * @param {URLSearchParams} searchParams - The search params to append to the URL
 * @returns {string} - The joined URL
 */
function url_join(baseURL, paths, searchParams) {
  const url = new URL(baseURL);
  const pathnames = [...url.pathname.split("/").filter((x) => x), ...paths];

  url.pathname = pathnames.join("/").replace(/\/{2,}/g, "/");

  for (const [key, value] of searchParams) {
    url.searchParams.append(key, value);
  }
  return url.toString();
}

class APIClient {
  constructor(baseURL) {
    this.client = axios.create({
      baseURL: baseURL,
      headers: {
        "Content-Type": "application/json",
      },
    });
  }

  synthesizeTTSUrl({
    text,
    spk = "female2",
    style = "chat",
    temperature = 0.3,
    top_p = 0.7,
    top_k = 20,
    seed = 34060637,
    format = "mp3",
    prompt1 = "",
    prompt2 = "",
    prefix = "",
    bs = 8,
    thr = 100,
    no_cache = false,
    stream = false,
    model = "chat-tts",
  }) {
    const params = new URLSearchParams({
      text,
      spk,
      style,
      temperature,
      top_p,
      top_k,
      seed,
      format,
      prompt1,
      prompt2,
      prefix,
      bs,
      thr,
      no_cache,
      stream,
      model,
    });
    // return `${this.client.defaults.baseURL}v1/tts?${params.toString()}`;

    // const url = new URL("/v1/tts", this.client.defaults.baseURL);
    // url.search = params.toString();
    // return url.toString();

    return url_join(this.client.defaults.baseURL, ["v1/tts"], params);
  }

  async synthesizeTTS({
    text,
    spk = "female2",
    style = "chat",
    temperature = 0.3,
    top_p = 0.7,
    top_k = 20,
    seed = 34060637,
    format = "mp3",
    prompt1 = "",
    prompt2 = "",
    prefix = "",
    bs = 8,
    thr = 100,
  }) {
    try {
      const response = await this.client.get("/v1/tts", {
        params: {
          text,
          spk,
          style,
          temperature,
          top_p,
          top_k,
          seed,
          format,
          prompt1,
          prompt2,
          prefix,
          bs,
          thr,
        },
        responseType: "blob", // Important for handling binary data
      });
      return response.data;
    } catch (error) {
      console.error("Error synthesizing TTS:", error);
      throw error;
    }
  }

  /**
   *
   * @param {string} ssml - The SSML text to synthesize
   * @param {string} format - The format of the synthesized audio (e.g. "mp3")
   */
  async synthesizeSSML({ ssml, format = "mp3" }) {
    try {
      const response = await this.client.post(
        "/v1/ssml",
        {
          ssml,
          format,
          batch_size: 4, // 默认 最大不超过 6gb
          // batch_size: 12, // 最长 500 token 的话大概 4gb
          // batch_size: 30, // 最长 500 token 的话大概 16gb
        },
        {
          responseType: "blob", // Important for handling binary data
        }
      );
      return response.data;
    } catch (error) {
      console.error("Error synthesizing SSML:", error);
      throw error;
    }
  }

  async openaiSpeechAPI({
    input,
    model = "chattts-4w",
    voice = "female2",
    response_format = "mp3",
    speed = 1,
  }) {
    try {
      const response = await this.client.post(
        "/v1/audio/speech",
        {
          input,
          model,
          voice,
          response_format,
          speed,
        },
        {
          responseType: "blob", // Important for handling binary data
        }
      );
      return response.data;
    } catch (error) {
      console.error("Error generating speech audio:", error);
      throw error;
    }
  }

  async refineText({
    text,
    prompt = "[oral_2][laugh_0][break_6]",
    seed = -1,
    top_P = 0.7,
    top_K = 20,
    temperature = 0.7,
    repetition_penalty = 1.0,
    max_new_token = 384,
  }) {
    try {
      const response = await this.client.post("/v1/prompt/refine", {
        text,
        prompt,
        seed,
        top_P,
        top_K,
        temperature,
        repetition_penalty,
        max_new_token,
      });
      return response.data;
    } catch (error) {
      console.error("Error refining text:", error);
      throw error;
    }
  }

  async listStyles() {
    try {
      const response = await this.client.get("/v1/styles/list");
      return response.data;
    } catch (error) {
      console.error("Error listing styles:", error);
      throw error;
    }
  }

  async listSpeakers() {
    try {
      const response = await this.client.get("/v1/speakers/list");
      /**
       * @type {{message:string,data:{items:any[],limit:number,offset:number,total:number}}}
       */
      const ret = response.data;
      return ret;
    } catch (error) {
      console.error("Error listing speakers:", error);
      throw error;
    }
  }

  async getAudioFormats() {
    try {
      const response = await this.client.get(
        "/v1/audio_formats",
        {},
        {
          responseType: "json",
        }
      );
      return response.data;
    } catch (error) {
      console.error("Error get audio formats:", error);
      return { data: ["mp3", "wav"] };
    }
  }

  async createSpeaker({ name, seed, describe, gender, tensor }) {
    try {
      const response = await this.client.post("/v1/speaker/create", {
        name,
        seed,
        gender,
        describe,
        tensor,
      });
      return response.data;
    } catch (error) {
      console.error("Error creating speaker:", error);
      throw error;
    }
  }

  async updateSpeakers({ speakers }) {
    try {
      const response = await this.client.post("/v1/speakers/update", {
        speakers,
      });
      return response.data;
    } catch (error) {
      console.error("Error updating speakers:", error);
      throw error;
    }
  }

  async googleTTTS(payload) {
    try {
      const response = await this.client.post("/v1/text:synthesize", payload, {
        responseType: "json",
      });
      return response.data;
    } catch (error) {
      console.error("Error synthesizing TTS with Google:", error);
      throw error;
    }
  }
}

export const client = new APIClient(
  localStorage.getItem("__chattts_playground_api_base__") ||
    `${window.location.origin}`
);
