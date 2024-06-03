import { render } from "preact";
import { html, create, styled } from "./misc.mjs";

import { pages } from "./pages.mjs";

const useStore = create((set, get) => ({
  page: Object.keys(pages).includes(location.hash.slice(1))
    ? location.hash.slice(1)
    : Object.keys(pages)[0],
  setPage: (page) => {
    set({ page });
    location.hash = page;
  },
}));

const NotFound = () => html`<div>Not Found</div>`;

const Content = () => {
  const { page } = useStore();
  const Page = pages[page];
  return Page ? html`<${Page} />` : html`<${NotFound} />`;
};

const NavButton = styled.button`
  background-color: transparent;
  border: none;
  color: white;
  cursor: pointer;
  font-size: 1rem;
  padding: 0.5rem 1rem;
  border-bottom: 1px solid transparent;

  &:hover {
    background-color: #444;
  }

  &.--active {
    border-bottom: 1px solid blue;
  }
`;

const HeaderNav = styled.nav`
  display: flex;
  padding: 4px 8px;
  background-color: #333;

  gap: 8px;

  user-select: none;

  .nav-icon {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .nav-title {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .divider {
    flex: 1;
  }
`;

const PageNav = () => {
  const { setPage, page: current } = useStore();
  return html`
    <${HeaderNav}>
      <div class="nav-icon">🗣️</div>
      <div class="nav-title">ChatTTS Forge Playground</div>
      ${Object.keys(pages).map(
        (page) =>
          html`
            <${NavButton}
              onClick=${() => setPage(page)}
              className=${current === page ? "--active" : ""}
            >
              ${page}
            <//>
          `
      )}
      <div class="divider"></div>
      <${NavButton}
        onClick=${() => {
          window.open("https://github.com/lenML/ChatTTS-Forge", "_blank");
        }}
      >
        github
      <//>
    <//>
  `;
};

const AppContent = styled.div`
  padding: 8px;
  flex: 1;
  overflow: auto;
`;

const App = () => {
  const { page } = useStore();
  return html`
    <${PageNav} />
    <${AppContent} className="pg-scrollbar">
      <${Content} />
    <//>
  `;
};

render(html`<${App} />`, document.getElementById("app"));
