import AppShell from "./app/App";
import AuthGate from "./app/AuthGate";

export default function App() {
  return (
    <AuthGate>
      <AppShell />
    </AuthGate>
  );
}
