import { NextRequest } from "next/server";

import { proxyToBackend } from "@/shared/api/proxy";

type RouteContext = { params: Promise<{ path: string[] }> };

async function handle(request: NextRequest, context: RouteContext) {
  const { path } = await context.params;
  const backendPath = `/${path.join("/")}`;
  return proxyToBackend(request, backendPath);
}

export const GET = handle;
export const POST = handle;
export const PUT = handle;
export const PATCH = handle;
export const DELETE = handle;
