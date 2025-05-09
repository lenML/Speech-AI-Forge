import { SttClient } from "./SttClient.mjs";
import { writeFileSync } from "fs";

const client = new SttClient("http://localhost:7870");

const [input_file = "./tests/test_inputs/chattts_out1.wav"] =
  process.argv.slice(2);

const result = await client.transcribe(input_file, {
  model: "whisper",
  // language: "en",
  format: "vtt",
  highlight_words: true,
});

console.log("Transcript:", result.data.text);

// 保存到 demo_stt_output_{date}.json
const date = new Date().toISOString().slice(0, 10);
const output_file = `demo_stt_output_${date}.json`;
writeFileSync(output_file, JSON.stringify(result.data, null, 2));
console.log("Saved to", output_file);
