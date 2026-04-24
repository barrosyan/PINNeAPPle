import { useEffect, useRef, useCallback, useState } from "react";
import type { WSMessage } from "@/types";

interface UseWebSocketOptions {
  onMessage?: (msg: WSMessage) => void;
  onOpen?:    () => void;
  onClose?:   () => void;
  enabled?:   boolean;
}

export function useWebSocket(wsRunId: string | null, opts: UseWebSocketOptions = {}) {
  const wsRef         = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const optsRef       = useRef(opts);
  optsRef.current     = opts;

  const connect = useCallback(() => {
    if (!wsRunId) return;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const host     = window.location.host;
    const url      = `${protocol}://${host}/ws/training/${wsRunId}/`;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      optsRef.current.onOpen?.();
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data) as WSMessage;
        optsRef.current.onMessage?.(msg);
      } catch { /* ignore parse errors */ }
    };

    ws.onclose = () => {
      setConnected(false);
      optsRef.current.onClose?.();
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [wsRunId]);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  const send = useCallback((data: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  useEffect(() => {
    if (opts.enabled !== false && wsRunId) {
      connect();
    }
    return disconnect;
  }, [wsRunId, opts.enabled, connect, disconnect]);

  return { connected, send, disconnect };
}
