import { Injectable, Logger } from '@nestjs/common';
import { Registry, Gauge, Counter, Histogram } from 'prom-client';

@Injectable()
export class JpmorganMetricsService {
  private readonly logger = new Logger(JpmorganMetricsService.name);
  public readonly register: Registry;

  // Gauges
  private readonly balanceGauge: Gauge<string>;
  private readonly lastSuccessGauge: Gauge<string>;
  private readonly tokenExpiryGauge: Gauge<string>;

  // Counters
  private readonly apiCallsCounter: Counter<string>;
  private readonly apiErrorsCounter: Counter<string>;
  private readonly tokenRefreshCounter: Counter<string>;

  // Histograms
  private readonly apiDurationHistogram: Histogram<string>;
  private readonly tokenAcquisitionHistogram: Histogram<string>;

  constructor() {
    this.register = new Registry();

    // Account Balance Gauge
    this.balanceGauge = new Gauge({
      name: 'jpm_account_balance',
      help: 'JPMorgan account balance by account ID and currency',
      labelNames: ['accountId', 'accountName', 'accountType', 'currency'],
      registers: [this.register],
    });

    // Last Successful API Call
    this.lastSuccessGauge = new Gauge({
      name: 'jpm_api_last_success_timestamp',
      help: 'Last successful JPMorgan API call (unix seconds)',
      labelNames: ['endpoint'],
      registers: [this.register],
    });

    // Token Expiry Time
    this.tokenExpiryGauge = new Gauge({
      name: 'jpm_token_expiry_timestamp',
      help: 'JPMorgan OAuth2 token expiry time (unix seconds)',
      registers: [this.register],
    });

    // API Calls Counter
    this.apiCallsCounter = new Counter({
      name: 'jpm_api_calls_total',
      help: 'Total number of JPMorgan API calls',
      labelNames: ['endpoint', 'status'],
      registers: [this.register],
    });

    // API Errors Counter
    this.apiErrorsCounter = new Counter({
      name: 'jpm_api_errors_total',
      help: 'Total number of JPMorgan API errors',
      labelNames: ['endpoint', 'error_type'],
      registers: [this.register],
    });

    // Token Refresh Counter
    this.tokenRefreshCounter = new Counter({
      name: 'jpm_token_refresh_total',
      help: 'Total number of OAuth2 token refreshes',
      labelNames: ['status'],
      registers: [this.register],
    });

    // API Duration Histogram
    this.apiDurationHistogram = new Histogram({
      name: 'jpm_api_duration_seconds',
      help: 'JPMorgan API call duration in seconds',
      labelNames: ['endpoint'],
      buckets: [0.1, 0.5, 1, 2, 5, 10],
      registers: [this.register],
    });

    // Token Acquisition Duration
    this.tokenAcquisitionHistogram = new Histogram({
      name: 'jpm_token_acquisition_duration_seconds',
      help: 'OAuth2 token acquisition duration in seconds',
      buckets: [0.1, 0.5, 1, 2, 5],
      registers: [this.register],
    });

    this.logger.log('Prometheus metrics initialized');
  }

  // Update account balance
  updateBalance(
    accountId: string,
    accountName: string,
    accountType: string,
    currency: string,
    balance: number,
  ): void {
    this.balanceGauge
      .labels(accountId, accountName, accountType, currency)
      .set(balance);
  }

  // Record successful API call
  recordApiSuccess(endpoint: string): void {
    this.apiCallsCounter.labels(endpoint, 'success').inc();
    this.lastSuccessGauge.labels(endpoint).set(Math.floor(Date.now() / 1000));
  }

  // Record API error
  recordApiError(endpoint: string, errorType: string): void {
    this.apiCallsCounter.labels(endpoint, 'error').inc();
    this.apiErrorsCounter.labels(endpoint, errorType).inc();
  }

  // Record API duration
  recordApiDuration(endpoint: string, durationSeconds: number): void {
    this.apiDurationHistogram.labels(endpoint).observe(durationSeconds);
  }

  // Record token refresh
  recordTokenRefresh(success: boolean): void {
    this.tokenRefreshCounter.labels(success ? 'success' : 'failure').inc();
  }

  // Record token acquisition duration
  recordTokenAcquisitionDuration(durationSeconds: number): void {
    this.tokenAcquisitionHistogram.observe(durationSeconds);
  }

  // Update token expiry
  updateTokenExpiry(expiryTimestamp: number): void {
    this.tokenExpiryGauge.set(expiryTimestamp);
  }

  // Get metrics
  async getMetrics(): Promise<string> {
    return this.register.metrics();
  }

  // Get content type
  getContentType(): string {
    return this.register.contentType;
  }
}
