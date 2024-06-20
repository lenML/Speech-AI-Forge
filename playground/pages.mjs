import { TTSPage } from "./tts.page.mjs";
import { SSMLPage } from "./ssml.page.mjs";
import { SpeakerPage } from "./speakers.page.mjs";
import { OpenaiPage } from "./openai.page.mjs";
import { GooglePage } from "./google.page.mjs";
import { AudioCreation } from "./AudioCreation.page.mjs";
import { StreamPage } from "./stream.page.mjs";

export const pages = {
  tts: TTSPage,
  ssml: SSMLPage,
  speakers: SpeakerPage,
  openai: OpenaiPage,
  google: GooglePage,
  AudioCreation: AudioCreation,
  stream: StreamPage,
};
