import { BrowserRouter, Route, Routes } from "react-router-dom";

import { Layout } from "./components/Layout";
import { HomePage } from "./pages/HomePage";
import { UploadPage } from "./pages/UploadPage";
import { ResultsPage } from "./pages/ResultsPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<HomePage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/cases/:caseId/jobs/:jobId" element={<ResultsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
