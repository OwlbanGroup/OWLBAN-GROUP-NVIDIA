import { Injectable, Logger } from '@nestjs/common';
import { Counter, Gauge, Histogram, Registry } from 'prom-client';
import { PaymentType } from '../enums/payment-type.enum';
import { PaymentStatus } from '../enums/payment-status.enum';
import { PaymentDirection } from '../enums/payment-direction.enum';

@Injectable()
export class PaymentMetricsService {
  private readonly logger = new Logger(PaymentMetricsService.name);
  private readonly registry: Registry;

  // Counters
  private readonly paymentsCreatedCounter: Counter;
  private readonly paymentsApprovedCounter: Counter;
  private readonly paymentsSubmittedCounter: Counter;
  private readonly paymentsCompletedCounter: Counter;
  private readonly paymentsFailedCounter: Counter;
  private readonly paymentsCancelledCounter: Counter;

  // Gauges
  private readonly paymentsInProgressGauge: Gauge;
  private readonly paymentsPendingApprovalGauge: Gauge;
  private readonly totalPaymentAmountGauge: Gauge;

  // Histograms
  private readonly paymentProcessingDuration: Histogram;
  private readonly paymentApprovalDuration: Histogram;
  private readonly paymentAmountHistogram: Histogram;

  constructor() {
    this.registry = new Registry();

    // Initialize Counters
    this.paymentsCreatedCounter = new Counter({
      name: 'payments_created_total',
      help: 'Total number of payments created',
      labelNames: ['type', 'direction', 'currency'],
      registers: [this.registry],
    });

    this.paymentsApprovedCounter = new Counter({
      name: 'payments_approved_total',
      help: 'Total number of payments approved',
      labelNames: ['type', 'direction', 'currency'],
      registers: [this.registry],
    });

    this.paymentsSubmittedCounter = new Counter({
      name: 'payments_submitted_total',
      help: 'Total number of payments submitted to JPMorgan',
      labelNames: ['type', 'direction', 'currency'],
      registers: [this.registry],
    });

    this.paymentsCompletedCounter = new Counter({
      name: 'payments_completed_total',
      help: 'Total number of payments completed successfully',
      labelNames: ['type', 'direction', 'currency'],
      registers: [this.registry],
    });

    this.paymentsFailedCounter = new Counter({
      name: 'payments_failed_total',
      help: 'Total number of payments that failed',
      labelNames: ['type', 'direction', 'currency', 'error_code'],
      registers: [this.registry],
    });

    this.paymentsCancelledCounter = new Counter({
      name: 'payments_cancelled_total',
      help: 'Total number of payments cancelled',
      labelNames: ['type', 'direction', 'currency'],
      registers: [this.registry],
    });

    // Initialize Gauges
    this.paymentsInProgressGauge = new Gauge({
      name: 'payments_in_progress',
      help: 'Number of payments currently in progress',
      labelNames: ['type', 'status'],
      registers: [this.registry],
    });

    this.paymentsPendingApprovalGauge = new Gauge({
      name: 'payments_pending_approval',
      help: 'Number of payments pending approval',
      labelNames: ['type'],
      registers: [this.registry],
    });

    this.totalPaymentAmountGauge = new Gauge({
      name: 'total_payment_amount_cents',
      help: 'Total amount of payments in cents',
      labelNames: ['type', 'direction', 'currency', 'status'],
      registers: [this.registry],
    });

    // Initialize Histograms
    this.paymentProcessingDuration = new Histogram({
      name: 'payment_processing_duration_seconds',
      help: 'Duration of payment processing in seconds',
      labelNames: ['type', 'status'],
      buckets: [0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
      registers: [this.registry],
    });

    this.paymentApprovalDuration = new Histogram({
      name: 'payment_approval_duration_seconds',
      help: 'Duration from creation to approval in seconds',
      labelNames: ['type'],
      buckets: [60, 300, 600, 1800, 3600, 7200, 14400, 28800, 86400],
      registers: [this.registry],
    });

    this.paymentAmountHistogram = new Histogram({
      name: 'payment_amount_cents',
      help: 'Distribution of payment amounts in cents',
      labelNames: ['type', 'direction', 'currency'],
      buckets: [100, 1000, 10000, 100000, 1000000, 10000000, 100000000],
      registers: [this.registry],
    });

    this.logger.log('Payment metrics initialized');
  }

  /**
   * Record payment creation
   */
  recordPaymentCreated(
    type: PaymentType,
    direction: PaymentDirection,
    currency: string,
    amountCents: number,
  ): void {
    this.paymentsCreatedCounter.inc({
      type,
      direction,
      currency,
    });

    this.paymentAmountHistogram.observe(
      { type, direction, currency },
      amountCents,
    );

    this.logger.debug(`Payment created: ${type} ${direction} ${amountCents} ${currency}`);
  }

  /**
   * Record payment approval
   */
  recordPaymentApproved(
    type: PaymentType,
    direction: PaymentDirection,
    currency: string,
    approvalDurationSeconds: number,
  ): void {
    this.paymentsApprovedCounter.inc({
      type,
      direction,
      currency,
    });

    this.paymentApprovalDuration.observe(
      { type },
      approvalDurationSeconds,
    );

    this.logger.debug(`Payment approved: ${type} (${approvalDurationSeconds}s)`);
  }

  /**
   * Record payment submission
   */
  recordPaymentSubmitted(
    type: PaymentType,
    direction: PaymentDirection,
    currency: string,
  ): void {
    this.paymentsSubmittedCounter.inc({
      type,
      direction,
      currency,
    });

    this.logger.debug(`Payment submitted: ${type} ${direction}`);
  }

  /**
   * Record payment completion
   */
  recordPaymentCompleted(
    type: PaymentType,
    direction: PaymentDirection,
    currency: string,
    processingDurationSeconds: number,
  ): void {
    this.paymentsCompletedCounter.inc({
      type,
      direction,
      currency,
    });

    this.paymentProcessingDuration.observe(
      { type, status: 'completed' },
      processingDurationSeconds,
    );

    this.logger.debug(`Payment completed: ${type} (${processingDurationSeconds}s)`);
  }

  /**
   * Record payment failure
   */
  recordPaymentFailed(
    type: PaymentType,
    direction: PaymentDirection,
    currency: string,
    errorCode: string,
    processingDurationSeconds: number,
  ): void {
    this.paymentsFailedCounter.inc({
      type,
      direction,
      currency,
      error_code: errorCode,
    });

    this.paymentProcessingDuration.observe(
      { type, status: 'failed' },
      processingDurationSeconds,
    );

    this.logger.debug(`Payment failed: ${type} ${errorCode}`);
  }

  /**
   * Record payment cancellation
   */
  recordPaymentCancelled(
    type: PaymentType,
    direction: PaymentDirection,
    currency: string,
  ): void {
    this.paymentsCancelledCounter.inc({
      type,
      direction,
      currency,
    });

    this.logger.debug(`Payment cancelled: ${type}`);
  }

  /**
   * Update payments in progress gauge
   */
  updatePaymentsInProgress(
    type: PaymentType,
    status: PaymentStatus,
    count: number,
  ): void {
    this.paymentsInProgressGauge.set(
      { type, status },
      count,
    );
  }

  /**
   * Update payments pending approval gauge
   */
  updatePaymentsPendingApproval(type: PaymentType, count: number): void {
    this.paymentsPendingApprovalGauge.set({ type }, count);
  }

  /**
   * Update total payment amount gauge
   */
  updateTotalPaymentAmount(
    type: PaymentType,
    direction: PaymentDirection,
    currency: string,
    status: PaymentStatus,
    totalAmountCents: number,
  ): void {
    this.totalPaymentAmountGauge.set(
      { type, direction, currency, status },
      totalAmountCents,
    );
  }

  /**
   * Get metrics in Prometheus format
   */
  async getMetrics(): Promise<string> {
    return this.registry.metrics();
  }

  /**
   * Get metrics registry
   */
  getRegistry(): Registry {
    return this.registry;
  }

  /**
   * Reset all metrics (useful for testing)
   */
  resetMetrics(): void {
    this.registry.resetMetrics();
    this.logger.debug('Payment metrics reset');
  }
}
