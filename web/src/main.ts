import { mount } from "svelte";
import { inject } from "@vercel/analytics";
import App from "./App.svelte";
import { DEMO } from "./lib/config";
import "./app.css";

if (import.meta.env.VITE_VERCEL_ENV === "production" && !DEMO) {
  inject({ mode: "production" });
}

export default mount(App, { target: document.getElementById("app")! });
