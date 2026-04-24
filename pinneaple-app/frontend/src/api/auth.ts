import api from "./client";

export interface AuthUser {
  id:          number;
  username:    string;
  email:       string;
  first_name:  string;
  last_name:   string;
  date_joined: string;
}

export interface AuthTokens {
  access:  string;
  refresh: string;
}

export interface LoginResponse extends AuthTokens {
  user?: AuthUser;
}

export const login = (username: string, password: string) =>
  api.post<AuthTokens>("/auth/login/", { username, password }).then((r) => r.data);

export const register = (data: {
  username:    string;
  password:    string;
  email?:      string;
  first_name?: string;
  last_name?:  string;
}) =>
  api.post<{ user: AuthUser } & AuthTokens>("/auth/register/", data).then((r) => r.data);

export const refreshToken = (refresh: string) =>
  api.post<{ access: string }>("/auth/token/refresh/", { refresh }).then((r) => r.data);

export const logout = (refresh: string) =>
  api.post("/auth/logout/", { refresh }).then((r) => r.data);

export const getMe = () =>
  api.get<AuthUser>("/auth/me/").then((r) => r.data);

export const updateMe = (data: Partial<Pick<AuthUser, "first_name" | "last_name" | "email">>) =>
  api.patch<AuthUser>("/auth/me/", data).then((r) => r.data);

export const changePassword = (current_password: string, new_password: string) =>
  api.post<AuthTokens & { detail: string }>(
    "/auth/change-password/",
    { current_password, new_password }
  ).then((r) => r.data);
