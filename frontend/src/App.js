import "@/App.css";
import { BrowserRouter, Routes, Route, useLocation, Navigate } from "react-router-dom";

import Login from "@/pages/Login";
import Sender from "@/pages/Sender";
import Receiver from "@/pages/Receiver";
import AuthCallback from "@/pages/AuthCallback";
import { AuthProvider } from "@/lib/auth-context";

function AppRoutes() {
  const location = useLocation();
  // Synchronous check (must run during render, NOT useEffect) to handle
  // the OAuth callback before any other auth check fires.
  if (location.hash?.includes("session_id=")) {
    return <AuthCallback />;
  }
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/sender" replace />} />
      <Route path="/login" element={<Login />} />
      <Route path="/sender" element={<Sender />} />
      <Route path="/receiver" element={<Receiver />} />
      <Route path="*" element={<Navigate to="/sender" replace />} />
    </Routes>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
