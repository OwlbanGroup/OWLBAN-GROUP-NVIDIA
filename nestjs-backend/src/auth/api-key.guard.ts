import {
  CanActivate,
  ExecutionContext,
  Injectable,
  UnauthorizedException,
  ForbiddenException,
} from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { ApiKeys } from './api-key-roles.config';
import { ROLES_KEY } from './auth.decorator';
import { Role } from './roles.enum';

@Injectable()
export class ApiKeyGuard implements CanActivate {
  constructor(private reflector: Reflector) {}

  canActivate(context: ExecutionContext): boolean {
    const requiredRoles =
      this.reflector.get<Role[]>(ROLES_KEY, context.getHandler()) || [];

    const request = context.switchToHttp().getRequest();
    const apiKey = request.headers['x-api-key'] as string | undefined;

    if (!apiKey) {
      throw new UnauthorizedException('Missing API key');
    }

    const role = ApiKeys[apiKey];
    if (!role) {
      throw new UnauthorizedException('Invalid API key');
    }

    if (requiredRoles.length && !requiredRoles.includes(role)) {
      throw new ForbiddenException('Insufficient role');
    }

    // Attach role for logging / downstream use
    request.userRole = role;

    return true;
  }
}
