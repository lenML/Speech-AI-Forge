import { client } from "./client.mjs";
import { create } from "./misc.mjs";

export const useGlobalStore = create((set, get) => ({
  speakers: [],
  styles: [],
}));

window.addEventListener("load", async () => {
  const styles = await client.listStyles();
  const speakers = await client.listSpeakers();
  console.log("styles:", styles);
  console.log("speakers:", speakers);
  useGlobalStore.set({
    styles: styles.data,
    speakers: speakers.data,
  });
});
