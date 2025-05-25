export interface ITTSParams {
  text: string; // 文本内容
  model?: string; // 模型名称，默认为 "chat-tts"

  spk?: string; // 说话人
  style?: string; // 风格
  temperature?: number; // 温度
  top_p?: number; // Top-p 参数
  top_k?: number; // Top-k 参数
  seed?: number; // 随机种子
  format?: string; // 格式
  bitrate?: number; // 比特率
  bs?: boolean; // 是否使用某功能（例子）
  thr?: number; // 阈值
  eos?: string; // 结束标点，默认为 "。"
  enhance?: boolean; // 是否增强，默认为 false
  denoise?: boolean; // 是否降噪，默认为 false
  speed?: number; // 速度，默认为 1
  pitch?: number; // 音调，默认为 0
  volume_gain?: number; // 音量增益，默认为 0
  stream?: boolean; // 是否流式，默认为 false
  chunk_size?: number; // 数据块大小，默认为 64
  no_cache?: boolean; // 是否禁用缓存，默认为 false
}

export interface ITTSParamsV2 {
  text?: string;
  texts?: string[];
  ssml?: string;
  adjust?: {
    pitch?: number;
    speed_rate?: number;
    volume_gain_db?: number;
    normalize?: boolean;
    headroom?: number;
  };
  encoder?: {
    bitrate?: string; // e.g., "64k"
    format?: string; // e.g., "mp3"
    acodec?: string;
  };
  enhance?: {
    enabled?: boolean;
    model?: string; // e.g., "resemble-enhance"
    nfe?: number;
    solver?: string; // e.g., "midpoint"
    lambd?: number;
    tau?: number;
  };
  infer?: {
    batch_size?: number;
    spliter_threshold?: number;
    eos?: string;
    seed?: number;
    stream?: boolean;
    stream_chunk_size?: number;
    no_cache?: boolean;
    sync_gen?: boolean;
  };
  vc?: {
    enabled?: boolean;
    mid?: string; // e.g., "open-voice"
    emotion?: string;
    tau?: number;
  };
  tn?: {
    enabled?: string[];
    disabled?: string[];
  };
  tts?: {
    mid?: string; // e.g., "chat-tts"
    style?: string;
    temperature?: number;
    top_p?: number;
    top_k?: number;
    prompt?: string;
    prompt1?: string;
    prompt2?: string;
    prefix?: string;
    emotion?: string;
  };
  spk?: {
    from_spk_id?: string;
    from_spk_name?: string;
    from_ref?: {
      wav_b64: string;
      text: string;
    };
  };
}

export namespace Speaker {
  export interface AudioReference {
    wav_b64: string;
    text: string;
    emotion?: string;
  }

  export interface CreateSpeaker {
    name: string;
    gender?: string;
    author?: string;
    desc?: string;
    version?: string;
    wavs?: AudioReference[];
    save_file?: boolean;
  }

  export interface UpdateSpeaker {
    json?: Record<string, any>;
  }

  export interface SpeakerDetail {
    id: string;
    with_emb?: boolean;
  }

  export interface SpeakersUpdate {
    speakers: Record<string, any>[];
  }

  export interface SpkListParams {
    detailed?: boolean;
    offset?: number;
    limit?: number;
  }

  export interface Speaker {
    id: string;
    name: string;
    gender?: string;
    author?: string;
    desc?: string;
    version?: string;
    [key: string]: any;
  }

  export interface PaginatedResponse<T> {
    items: T[];
    offset: number;
    limit: number;
    total: number;
  }

  export interface BaseResponse<T = any> {
    // code: number;
    message: string;
    data: T;
  }
}
