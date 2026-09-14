import { BrowserRouter, Route, Routes } from "react-router-dom";

import { Layout } from "./components/Layout";
import { HomePage } from "./pages/HomePage";
import { UploadPage } from "./pages/UploadPage";
import { ResultsPage } from "./pages/ResultsPage";
import { ViewerPage } from "./pages/ViewerPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<HomePage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/cases/:caseId/jobs/:jobId" element={<ResultsPage />} />
          <Route path="/cases/:caseId/viewer" element={<ViewerPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
