import { createRoot } from "react-dom";
import { html, create, styled } from "./misc.mjs";

import { pages } from "./pages.mjs";
import { ErrorBoundary } from "react-error-boundary";

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

  z-index: 10;
  box-shadow: rgba(0, 0, 0, 0.24) 0px 3px 8px;

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

function fallbackRender({ error, resetErrorBoundary }) {
  return html`<div role="alert">
    <p>Something went wrong:</p>
    <pre style=${{ color: "red" }}>${JSON.stringify(error.message)}</pre>
    <button onClick=${resetErrorBoundary}>reset</button>
  </div>`;
}

const App = () => {
  // NOTE: 不知道为啥... ErrorBoundary 莫名用不了...
  return html`
    <${PageNav} />
    <${AppContent} className="pg-scrollbar">
      <!-- <${ErrorBoundary} fallbackRender=${fallbackRender}> <//> -->
      <${Content} />
    <//>
  `;
};

const root = createRoot(document.getElementById("app"));
root.render(html`<${App} />`);
