import { client } from "./client.mjs";
import { html, create, styled } from "./misc.mjs";

import { OpenAI } from "openai";

import { useGlobalStore } from "./global.store.mjs";

const useStore = create((set, get) => ({
  // TODO
}));

const Container = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
`;

export const AudioCreation = () => {
  return html` <${Container}> TODO 🚧 AudioCreation page <//> `;
};
