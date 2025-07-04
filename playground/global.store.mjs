import { client } from "./client.mjs";
import { create } from "./misc.mjs";

/**
 * @type {() => {speakers:any[],styles:any[]}}
 */
export const useGlobalStore = create((set, get) => ({
  speakers: [],
  styles: [],
  formats: [],
}));

window.addEventListener("load", async () => {
  const styles = await client.listStyles();
  const speakers = await client.listSpeakers();
  const formats = await client.getAudioFormats();
  console.log("styles:", styles);
  console.log("speakers:", speakers);
  console.log("formats:", formats);
  useGlobalStore.set({
    styles: styles.data,
    speakers: speakers.data.items,
    formats: formats.data,
  });
});
