import { SAFClient } from "./client.mjs";
import fs from "fs";

const [ref_wav_filepath, ref_text] = process.argv.slice(2);
if (!ref_wav_filepath || !ref_text) {
  console.log(
    "Usage: node examples/javascript/tts_v2.mjs <ref_wav_filepath> <ref_text>"
  );
  process.exit(1);
}

const client = new SAFClient();

client
  .tts_v2({
    text: "你好，欢迎使用 Speech AI Forge 项目。",
    spk: {
      from_ref: {
        wav_b64: fs.readFileSync(ref_wav_filepath).toString("base64"),
        text: ref_text,
      },
    },
    // NOTE：或者使用mona内置音色
    // spk: {
    //   from_spk_name: "mona",
    // },

    // 控制音调调节
    adjuct: {
      pitch: 0,
      speed_rate: 1,
      volume_gain_db: 0,
      normalize: false,
      headroom: 1,
    },
    // 编码配置
    encoder: {
      bitrate: "64k",
      format: "mp3",
      acodec: "libmp3lame",
    },
    // 增强模型配置 （增强操作无缓存，可能导致生成延迟）
    enhance: {
      enabled: true,
      model: "resemble-enhance",
      nfe: 32,
      solver: "midpoint",
      lambd: 0.5,
      tau: 0.5,
    },
    // 推理引擎参数
    infer: {
      batch_size: 4,
      spliter_threshold: 100,
      eos: "。",
      seed: 42,
      stream: false,
      stream_chunk_size: 64,
      no_cache: false,
      sync_gen: false,
    },
    // 语音克隆参数 （准备废弃）
    vc: {
      enabled: false,
      mid: "open-voice",
      emotion: "default",
      tau: 0.3,
    },
    // TN配置，可以指定关闭某些TN，留空表示默认
    tn: {
      enabled: [],
      disabled: [],
    },
    // tts参数 指定模型和推理参数
    tts: {
      mid: "cosy-voice",
      style: "",
      temperature: 0.3,
      top_p: 0.7,
      top_k: 20,
      emotion: "default",

      // 以下参数将废弃
      prompt: "",
      prompt1: "",
      prompt2: "",
      prefix: "",
    },
  })
  .then((audio) => {
    fs.writeFileSync("./example_v2_full_out.mp3", Buffer.from(audio));
  });
