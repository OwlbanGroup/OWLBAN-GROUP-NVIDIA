import { Injectable, Logger } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Payment } from '../entities/payment.entity';
import { v4 as uuidv4 } from 'uuid';

interface IdempotencyRecord {
  key: string;
  paymentId: string;
  createdAt: Date;
  expiresAt: Date;
}

@Injectable()
export class IdempotencyService {
  private readonly logger = new Logger(IdempotencyService.name);
  private readonly cache = new Map<string, IdempotencyRecord>();
  private readonly DEFAULT_TTL = 24 * 60 * 60 * 1000; // 24 hours

  constructor(
    @InjectRepository(Payment)
    private paymentRepository: Repository<Payment>,
  ) {
    // Clean up expired entries every hour
    setInterval(() => this.cleanupExpired(), 60 * 60 * 1000);
  }

  /**
   * Generate a new idempotency key
   */
  generateKey(): string {
    return uuidv4();
  }

  /**
   * Check if an idempotency key has been used
   * Returns the payment ID if it exists, null otherwise
   */
  async checkKey(key: string): Promise<string | null> {
    // Check in-memory cache first
    const cached = this.cache.get(key);
    if (cached) {
      if (new Date() < cached.expiresAt) {
        this.logger.debug(`Idempotency key found in cache: ${key}`);
        return cached.paymentId;
      } else {
        // Expired, remove from cache
        this.cache.delete(key);
      }
    }

    // Check database
    const payment = await this.paymentRepository.findOne({
      where: { idempotencyKey: key },
      select: ['id', 'createdAt'],
    });

    if (payment) {
      // Check if payment is within TTL
      const age = Date.now() - payment.createdAt.getTime();
      if (age < this.DEFAULT_TTL) {
        this.logger.debug(`Idempotency key found in database: ${key}`);
        
        // Add to cache for faster future lookups
        this.cache.set(key, {
          key,
          paymentId: payment.id,
          createdAt: payment.createdAt,
          expiresAt: new Date(payment.createdAt.getTime() + this.DEFAULT_TTL),
        });

        return payment.id;
      }
    }

    return null;
  }

  /**
   * Register a new idempotency key with a payment ID
   */
  async registerKey(key: string, paymentId: string): Promise<void> {
    const now = new Date();
    const expiresAt = new Date(now.getTime() + this.DEFAULT_TTL);

    this.cache.set(key, {
      key,
      paymentId,
      createdAt: now,
      expiresAt,
    });

    this.logger.debug(`Registered idempotency key: ${key} -> ${paymentId}`);
  }

  /**
   * Remove an idempotency key (useful for testing or cleanup)
   */
  async removeKey(key: string): Promise<void> {
    this.cache.delete(key);
    this.logger.debug(`Removed idempotency key: ${key}`);
  }

  /**
   * Check if a key is valid (exists and not expired)
   */
  async isKeyValid(key: string): Promise<boolean> {
    const paymentId = await this.checkKey(key);
    return paymentId !== null;
  }

  /**
   * Get payment by idempotency key
   */
  async getPaymentByKey(key: string): Promise<Payment | null> {
    const payment = await this.paymentRepository.findOne({
      where: { idempotencyKey: key },
    });

    if (payment) {
      // Check if payment is within TTL
      const age = Date.now() - payment.createdAt.getTime();
      if (age < this.DEFAULT_TTL) {
        return payment;
      }
    }

    return null;
  }

  /**
   * Clean up expired entries from cache
   */
  private cleanupExpired(): void {
    const now = new Date();
    let cleaned = 0;

    for (const [key, record] of this.cache.entries()) {
      if (now > record.expiresAt) {
        this.cache.delete(key);
        cleaned++;
      }
    }

    if (cleaned > 0) {
      this.logger.debug(`Cleaned up ${cleaned} expired idempotency keys`);
    }
  }

  /**
   * Get cache statistics
   */
  getCacheStats() {
    return {
      size: this.cache.size,
      ttl: this.DEFAULT_TTL,
    };
  }

  /**
   * Clear all cached keys (useful for testing)
   */
  clearCache(): void {
    this.cache.clear();
    this.logger.debug('Cleared idempotency cache');
  }
}
