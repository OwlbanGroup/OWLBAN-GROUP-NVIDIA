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
import { Payment } from './payment.entity';
import { User } from '../../users/user.entity';
import { ApprovalStatus } from '../enums/approval-status.enum';

@Entity('payment_approvals')
@Index(['paymentId', 'status'])
@Index(['approverId'])
export class PaymentApproval {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  // Payment this approval belongs to
  @ManyToOne(() => Payment, { nullable: false, onDelete: 'CASCADE' })
  @JoinColumn({ name: 'paymentId' })
  payment: Payment;

  @Column({ nullable: false })
  paymentId: string;

  // User who needs to approve (or has approved/rejected)
  @ManyToOne(() => User, { nullable: false })
  @JoinColumn({ name: 'approverId' })
  approver: User;

  @Column({ nullable: false })
  approverId: string;

  // Approval status
  @Column({
    type: 'enum',
    enum: ApprovalStatus,
    default: ApprovalStatus.PENDING,
  })
  status: ApprovalStatus;

  // Approval level (for multi-level approvals)
  // Level 1 = first approver, Level 2 = second approver, etc.
  @Column({ type: 'int', default: 1 })
  level: number;

  // Order within the same level (for parallel approvals)
  @Column({ type: 'int', default: 1 })
  order: number;

  // Comments from the approver
  @Column({ type: 'text', nullable: true })
  comments: string;

  // Reason for rejection (if rejected)
  @Column({ type: 'text', nullable: true })
  rejectionReason: string;

  // IP address of the approver
  @Column({ nullable: true })
  ipAddress: string;

  // User agent of the approver
  @Column({ type: 'text', nullable: true })
  userAgent: string;

  // Timestamps
  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  @Column({ type: 'timestamp', nullable: true })
  approvedAt: Date;

  @Column({ type: 'timestamp', nullable: true })
  rejectedAt: Date;

  @Column({ type: 'timestamp', nullable: true })
  expiresAt: Date;

  // Metadata (flexible JSON field)
  @Column({ type: 'jsonb', nullable: true })
  metadata: any;

  // Helper method to check if approval is pending
  isPending(): boolean {
    return this.status === ApprovalStatus.PENDING;
  }

  // Helper method to check if approval is expired
  isExpired(): boolean {
    if (!this.expiresAt) return false;
    return new Date() > this.expiresAt;
  }

  // Helper method to check if approval can be acted upon
  canBeActedUpon(): boolean {
    return this.isPending() && !this.isExpired();
  }
}
