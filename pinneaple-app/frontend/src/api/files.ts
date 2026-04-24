import api from "./client";
import type { UploadedFile } from "@/types";

export const uploadFile = (file: File, projectId?: string) => {
  const fd = new FormData();
  fd.append("file", file);
  if (projectId) fd.append("project_id", projectId);
  return api.post<UploadedFile>("/files/upload/", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  }).then((r) => r.data);
};

export const getFiles = (projectId?: string) =>
  api.get<UploadedFile[]>("/files/", {
    params: projectId ? { project_id: projectId } : {},
  }).then((r) => r.data);

export const getFileData = (fileId: string) =>
  api.get<{ columns: string[]; data: Record<string, number[]>; rows: number }>(
    `/files/${fileId}/data/`
  ).then((r) => r.data);

export const deleteFile = (fileId: string) =>
  api.delete(`/files/${fileId}/`);
