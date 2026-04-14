"use client";

import { useState } from "react";

export default function Home() {
  const [players, setPlayers] = useState([]);

  const loadPlayers = async () => {
    const res = await fetch("https://afl-fantasy-analytics.onrender.com/players");
    const data = await res.json();
    setPlayers(data.slice(0, 20));
  };

  return (
    <main style={{ padding: 40 }}>
      <h1>AFL Fantasy Analytics</h1>

      <button onClick={loadPlayers}>
        Load Players
      </button>

      <ul>
        {players.map((p: any, i) => (
          <li key={i}>
            {p.name} - {p.predicted_points?.toFixed(1)}
          </li>
        ))}
      </ul>
    </main>
  );
}
