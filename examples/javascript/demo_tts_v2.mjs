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
  })
  .then((audio) => {
    fs.writeFileSync("./example_v2_out.mp3", Buffer.from(audio));
  });
