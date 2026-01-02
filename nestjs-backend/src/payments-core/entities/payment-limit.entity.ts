import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  CreateDateColumn,
  UpdateDateColumn,
  ManyToOne,
  JoinColumn,
  Index,
} from 'typeorm';
import { Organization } from '../../organizations/organization.entity';
import { User } from '../../users/user.entity';
import { PaymentType } from '../enums/payment-type.enum';
import { PaymentDirection } from '../enums/payment-direction.enum';

export enum LimitType {
  PER_TRANSACTION = 'PER_TRANSACTION',
  DAILY = 'DAILY',
  WEEKLY = 'WEEKLY',
  MONTHLY = 'MONTHLY',
}

export enum LimitScope {
  ORGANIZATION = 'ORGANIZATION',
  USER = 'USER',
  ROLE = 'ROLE',
}

@Entity('payment_limits')
@Index(['organizationId', 'isActive'])
@Index(['scope', 'isActive'])
export class PaymentLimit {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  // Organization this limit belongs to
  @ManyToOne(() => Organization, { nullable: false })
  @JoinColumn({ name: 'organizationId' })
  organization: Organization;

  @Column({ nullable: false })
  organizationId: string;

  // Limit name/description
  @Column({ nullable: false })
  name: string;

  @Column({ type: 'text', nullable: true })
  description: string;

  // Payment type this limit applies to (null = all types)
  @Column({
    type: 'enum',
    enum: PaymentType,
    nullable: true,
  })
  paymentType: PaymentType;

  // Payment direction this limit applies to (null = all directions)
  @Column({
    type: 'enum',
    enum: PaymentDirection,
    nullable: true,
  })
  direction: PaymentDirection;

  // Limit type (per transaction, daily, weekly, monthly)
  @Column({
    type: 'enum',
    enum: LimitType,
    nullable: false,
  })
  limitType: LimitType;

  // Limit scope (organization, user, role)
  @Column({
    type: 'enum',
    enum: LimitScope,
    nullable: false,
  })
  scope: LimitScope;

  // User this limit applies to (if scope is USER)
  @ManyToOne(() => User, { nullable: true })
  @JoinColumn({ name: 'userId' })
  user: User;

  @Column({ nullable: true })
  userId: string;

  // Role this limit applies to (if scope is ROLE)
  @Column({ nullable: true })
  role: string;

  // Limit amount in cents
  @Column({ type: 'bigint', nullable: false })
  limitAmountCents: number;

  // Currency code (ISO 4217)
  @Column({ length: 3, default: 'USD' })
  currency: string;

  // Whether this limit is active
  @Column({ default: true })
  isActive: boolean;

  // Priority (higher number = higher priority)
  // Used when multiple limits apply to determine which one to enforce
  @Column({ type: 'int', default: 0 })
  priority: number;

  // Whether to send alerts when approaching limit
  @Column({ default: true })
  alertEnabled: boolean;

  // Alert threshold percentage (e.g., 80 = alert at 80% of limit)
  @Column({ type: 'int', default: 80 })
  alertThresholdPercent: number;

  // Timestamps
  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  @Column({ type: 'timestamp', nullable: true })
  effectiveFrom: Date;

  @Column({ type: 'timestamp', nullable: true })
  effectiveUntil: Date;

  // User who created this limit
  @ManyToOne(() => User, { nullable: false })
  @JoinColumn({ name: 'createdById' })
  createdBy: User;

  @Column({ nullable: false })
  createdById: string;

  // Metadata (flexible JSON field)
  @Column({ type: 'jsonb', nullable: true })
  metadata: any;

  // Helper method to get limit amount in dollars
  getLimitAmountDollars(): number {
    return this.limitAmountCents / 100;
  }

  // Helper method to check if limit is currently effective
  isEffective(): boolean {
    if (!this.isActive) return false;
    
    const now = new Date();
    
    if (this.effectiveFrom && now < this.effectiveFrom) {
      return false;
    }
    
    if (this.effectiveUntil && now > this.effectiveUntil) {
      return false;
    }
    
    return true;
  }

  // Helper method to check if limit applies to a specific payment
  appliesToPayment(
    paymentType: PaymentType,
    direction: PaymentDirection,
    userId?: string,
    userRole?: string,
  ): boolean {
    if (!this.isEffective()) return false;

    // Check payment type
    if (this.paymentType && this.paymentType !== paymentType) {
      return false;
    }

    // Check direction
    if (this.direction && this.direction !== direction) {
      return false;
    }

    // Check scope
    if (this.scope === LimitScope.USER && this.userId !== userId) {
      return false;
    }

    if (this.scope === LimitScope.ROLE && this.role !== userRole) {
      return false;
    }

    return true;
  }

  // Helper method to calculate alert threshold amount
  getAlertThresholdAmountCents(): number {
    return Math.floor((this.limitAmountCents * this.alertThresholdPercent) / 100);
  }
}
