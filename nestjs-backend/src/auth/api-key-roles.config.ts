import { Role } from './roles.enum';

export const ApiKeys: Record<string, Role> = {
  // In production, load these from environment variables or a secrets store
  [process.env.DASHBOARD_ADMIN_API_KEY || 'admin-key-placeholder']: Role.ADMIN,
  [process.env.DASHBOARD_VIEWER_API_KEY || 'viewer-key-placeholder']: Role.VIEWER,
};
