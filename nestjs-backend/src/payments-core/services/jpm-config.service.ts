import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';

export enum JpmEnvironment {
  SANDBOX = 'sandbox',
  PRODUCTION = 'production',
}

@Injectable()
export class JpmConfigService {
  private readonly logger = new Logger(JpmConfigService.name);
  private readonly environment: JpmEnvironment;

  constructor(private configService: ConfigService) {
    const env = this.configService.get<string>('JPM_ENV', 'sandbox');
    this.environment = env === 'production' ? JpmEnvironment.PRODUCTION : JpmEnvironment.SANDBOX;
    
    this.logger.log(`JPMorgan environment: ${this.environment}`);
  }

  /**
   * Get the current JPMorgan environment
   */
  getEnvironment(): JpmEnvironment {
    return this.environment;
  }

  /**
   * Check if running in production environment
   */
  isProduction(): boolean {
    return this.environment === JpmEnvironment.PRODUCTION;
  }

  /**
   * Check if running in sandbox environment
   */
  isSandbox(): boolean {
    return this.environment === JpmEnvironment.SANDBOX;
  }

  /**
   * Get OAuth2 token URL
   */
  getTokenUrl(): string {
    return this.isProduction()
      ? this.configService.get<string>('JPM_PROD_TOKEN_URL', 'https://api.jpmorgan.com/oauth2/token')
      : this.configService.get<string>('JPM_SANDBOX_TOKEN_URL', 'https://sandbox.api.jpmorgan.com/oauth2/token');
  }

  /**
   * Get API base URL
   */
  getApiBaseUrl(): string {
    return this.isProduction()
      ? this.configService.get<string>('JPM_PROD_API_BASE_URL', 'https://api.jpmorgan.com')
      : this.configService.get<string>('JPM_SANDBOX_API_BASE_URL', 'https://sandbox.api.jpmorgan.com');
  }

  /**
   * Get OAuth2 client ID
   */
  getClientId(): string {
    return this.isProduction()
      ? this.configService.get<string>('JPM_PROD_CLIENT_ID', '')
      : this.configService.get<string>('JPM_SANDBOX_CLIENT_ID', '');
  }

  /**
   * Get OAuth2 client secret
   */
  getClientSecret(): string {
    return this.isProduction()
      ? this.configService.get<string>('JPM_PROD_CLIENT_SECRET', '')
      : this.configService.get<string>('JPM_SANDBOX_CLIENT_SECRET', '');
  }

  /**
   * Get OAuth2 scopes
   */
  getScopes(): string[] {
    const scopesStr = this.isProduction()
      ? this.configService.get<string>('JPM_PROD_SCOPES', 'payments:read payments:write')
      : this.configService.get<string>('JPM_SANDBOX_SCOPES', 'payments:read payments:write');
    
    return scopesStr.split(' ').filter(s => s.length > 0);
  }

  /**
   * Get mTLS certificate path (for production)
   */
  getCertificatePath(): string | null {
    if (!this.isProduction()) return null;
    return this.configService.get<string>('JPM_PROD_CERT_PATH') || null;
  }

  /**
   * Get mTLS private key path (for production)
   */
  getPrivateKeyPath(): string | null {
    if (!this.isProduction()) return null;
    return this.configService.get<string>('JPM_PROD_KEY_PATH') || null;
  }

  /**
   * Get HMAC signing key (if required)
   */
  getHmacKey(): string | null {
    return this.isProduction()
      ? this.configService.get<string>('JPM_PROD_HMAC_KEY') || null
      : this.configService.get<string>('JPM_SANDBOX_HMAC_KEY') || null;
  }

  /**
   * Get API timeout in milliseconds
   */
  getApiTimeout(): number {
    return this.configService.get<number>('JPM_API_TIMEOUT', 30000);
  }

  /**
   * Get max retry attempts
   */
  getMaxRetries(): number {
    return this.configService.get<number>('JPM_MAX_RETRIES', 3);
  }

  /**
   * Get retry delay in milliseconds
   */
  getRetryDelay(): number {
    return this.configService.get<number>('JPM_RETRY_DELAY', 1000);
  }

  /**
   * Check if HMAC signing is enabled
   */
  isHmacEnabled(): boolean {
    return this.getHmacKey() !== null;
  }

  /**
   * Check if mTLS is enabled
   */
  isMtlsEnabled(): boolean {
    return this.getCertificatePath() !== null && this.getPrivateKeyPath() !== null;
  }

  /**
   * Get complete configuration object
   */
  getConfig() {
    return {
      environment: this.environment,
      isProduction: this.isProduction(),
      isSandbox: this.isSandbox(),
      tokenUrl: this.getTokenUrl(),
      apiBaseUrl: this.getApiBaseUrl(),
      clientId: this.getClientId(),
      scopes: this.getScopes(),
      apiTimeout: this.getApiTimeout(),
      maxRetries: this.getMaxRetries(),
      retryDelay: this.getRetryDelay(),
      hmacEnabled: this.isHmacEnabled(),
      mtlsEnabled: this.isMtlsEnabled(),
    };
  }

  /**
   * Validate configuration
   */
  validateConfig(): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (!this.getClientId()) {
      errors.push(`Missing client ID for ${this.environment} environment`);
    }

    if (!this.getClientSecret()) {
      errors.push(`Missing client secret for ${this.environment} environment`);
    }

    if (this.isProduction()) {
      if (!this.getCertificatePath()) {
        errors.push('Missing certificate path for production environment');
      }

      if (!this.getPrivateKeyPath()) {
        errors.push('Missing private key path for production environment');
      }
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }
}
