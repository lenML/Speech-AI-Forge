/**
 *
 * @param {HTMLButtonElement} button
 * @param {string} url
 */
function download_speaker(button, url) {
  button.disabled = "disabled";
  button.value = "Downloading...";
  button.innerText = "Downloading...";

  var textarea = gradioApp().querySelector("#speaker_to_install textarea");
  textarea.value = url;
  updateInput(textarea);

  gradioApp().querySelector("#install_speaker_button").click();
}
