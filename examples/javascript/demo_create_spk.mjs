import fs from "fs";
import { SpkClient } from "./SpkClient.mjs";

const [audio_filepath, text = ""] = process.argv.slice(2);

if (!fs.existsSync(audio_filepath)) {
  console.error(`File not found: ${audio_filepath}`);
  process.exit(1);
}

const client = new SpkClient();

client
  .create({
    name: "my_spk",
    wavs: [
      {
        wav_b64: fs.readFileSync(audio_filepath).toString("base64"),
        text: text,
      },
    ],
  })
  .then((response) => {
    console.log(JSON.stringify(response.data, null, 2));
  })
  .catch((error) => console.error(error));
