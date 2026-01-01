import { plainToClass } from 'class-transformer';
import {
  IsEnum,
  IsNumber,
  IsString,
  validateSync,
  IsOptional,
  Min,
  Max,
} from 'class-validator';

enum Environment {
  Development = 'development',
  Production = 'production',
  Test = 'test',
  Staging = 'staging',
}

export class EnvironmentVariables {
  @IsEnum(Environment)
  @IsOptional()
  NODE_ENV: Environment = Environment.Development;

  @IsNumber()
  @Min(1)
  @Max(65535)
  PORT: number = 3000;

  // Database Configuration
  @IsString()
  DB_HOST: string;

  @IsNumber()
  @Min(1)
  @Max(65535)
  DB_PORT: number = 5432;

  @IsString()
  DB_USER: string;

  @IsString()
  DB_PASSWORD: string;

  @IsString()
  DB_NAME: string;

  @IsNumber()
  @IsOptional()
  @Min(1)
  @Max(100)
  DB_POOL_SIZE: number = 10;

  @IsNumber()
  @IsOptional()
  @Min(1000)
  DB_CONNECTION_TIMEOUT: number = 30000;

  // JWT Configuration
  @IsString()
  JWT_SECRET: string;

  @IsString()
  @IsOptional()
  JWT_EXPIRATION: string = '1h';

  // API Configuration
  @IsString()
  @IsOptional()
  API_PREFIX: string = 'api';

  @IsString()
  @IsOptional()
  API_VERSION: string = 'v1';

  // Rate Limiting
  @IsNumber()
  @IsOptional()
  @Min(1)
  THROTTLE_TTL: number = 60;

  @IsNumber()
  @IsOptional()
  @Min(1)
  THROTTLE_LIMIT: number = 10;

  // CORS Configuration
  @IsString()
  @IsOptional()
  CORS_ORIGIN: string = '*';

  // Logging
  @IsString()
  @IsOptional()
  LOG_LEVEL: string = 'info';

  // JPMorgan OAuth2 Configuration
  @IsString()
  JPM_CLIENT_ID: string;

  @IsString()
  JPM_CLIENT_SECRET: string;

  @IsString()
  @IsOptional()
  JPM_TOKEN_URL: string = 'https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token';

  @IsString()
  @IsOptional()
  JPM_SCOPE: string = 'jpm:payments:sandbox';

  @IsString()
  @IsOptional()
  JPM_API_BASE_URL: string = 'https://api-sandbox.payments.jpmorgan.com';
}

export function validate(config: Record<string, unknown>) {
  const validatedConfig = plainToClass(EnvironmentVariables, config, {
    enableImplicitConversion: true,
  });

  const errors = validateSync(validatedConfig, {
    skipMissingProperties: false,
  });

  if (errors.length > 0) {
    throw new Error(
      `Configuration validation error:\n${errors
        .map((err) => Object.values(err.constraints || {}).join(', '))
        .join('\n')}`,
    );
  }

  return validatedConfig;
}
