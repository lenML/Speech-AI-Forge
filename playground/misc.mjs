// import { h } from "preact";
import * as React from "react";
import htm from "htm";
import { bindReact } from "@quik-fe/stand";

import * as goober from "goober";

goober.setup(React.createElement);

export const html = htm.bind(React.createElement);
export const create = bindReact(React);

/**
 * @type {Record<string, Function>}
 */
export const styled = new Proxy(
  {},
  {
    get: (target, key) => goober.styled.call({}, key),
  }
);
