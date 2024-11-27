import { SAFClient } from "./client.mjs";
import fs from "fs";

const client = new SAFClient();

client
  .tts({
    text: "你好，欢迎使用 Speech AI Forge 项目。",
  })
  .then((audio) => {
    fs.writeFileSync("./example.mp3", Buffer.from(audio));
  });
