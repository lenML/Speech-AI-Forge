// Simulate an `input` DOM event for Gradio Textbox component. Needed after you edit its contents in javascript, otherwise your edits
// will only visible on web page and not sent to python.
function updateInput(target) {
  let e = new Event("input", { bubbles: true });
  Object.defineProperty(e, "target", { value: target });
  target.dispatchEvent(e);
}
