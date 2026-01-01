import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';

interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  scope: string;
}

@Injectable()
export class JpmorganTokenService {
  private readonly logger = new Logger(JpmorganTokenService.name);
  private cachedToken: string | null = null;
  private tokenExpiry: number | null = null;

  constructor(
    private readonly config: ConfigService,
    private readonly http: HttpService,
  ) {}

  async getAccessToken(): Promise<string> {
    const now = Date.now();

    // Return cached token if still valid
    if (this.cachedToken && this.tokenExpiry && now < this.tokenExpiry) {
      this.logger.debug('Using cached access token');
      return this.cachedToken;
    }

    this.logger.log('Fetching new access token from JPMorgan');

    try {
      const clientId = this.config.get<string>('JPM_CLIENT_ID');
      const clientSecret = this.config.get<string>('JPM_CLIENT_SECRET');
      const tokenUrl = this.config.get<string>('JPM_TOKEN_URL');
      const scope = this.config.get<string>('JPM_SCOPE');

      if (!clientId || !clientSecret || !tokenUrl) {
        throw new Error('JPMorgan OAuth2 credentials not configured');
      }

      const params = new URLSearchParams({
        client_id: clientId,
        client_secret: clientSecret,
        grant_type: 'client_credentials',
        scope: scope || 'jpm:payments:sandbox',
      });

      const response = await firstValueFrom(
        this.http.post<TokenResponse>(tokenUrl, params.toString(), {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
        }),
      );

      this.cachedToken = response.data.access_token;
      // Set expiry 30 seconds before actual expiry to avoid edge cases
      this.tokenExpiry = now + (response.data.expires_in - 30) * 1000;

      this.logger.log('Successfully obtained new access token');
      this.logger.debug(`Token expires in ${response.data.expires_in} seconds`);

      return this.cachedToken;
    } catch (error) {
      this.logger.error('Failed to obtain access token', error);
      throw new Error('Failed to authenticate with JPMorgan API');
    }
  }

  /**
   * Force token refresh on next request
   */
  invalidateToken(): void {
    this.logger.log('Invalidating cached token');
    this.cachedToken = null;
    this.tokenExpiry = null;
  }

  /**
   * Check if token is currently valid
   */
  isTokenValid(): boolean {
    const now = Date.now();
    return !!(this.cachedToken && this.tokenExpiry && now < this.tokenExpiry);
  }
}
